package dev.langchain4j.service;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.input.structured.StructuredPrompt;
import dev.langchain4j.model.input.structured.StructuredPromptProcessor;
import dev.langchain4j.model.moderation.Moderation;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.service.tool.ToolProviderResult;
import dev.langchain4j.spi.services.TokenStreamAdapter;

import java.lang.reflect.Array;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static dev.langchain4j.exception.IllegalConfigurationException.illegalConfiguration;
import static dev.langchain4j.internal.Exceptions.runtime;
import static dev.langchain4j.model.chat.request.ResponseFormatType.JSON;
import static dev.langchain4j.service.TypeUtils.typeHasRawClass;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;

class DefaultAiServices<T> extends AiServices<T> {

    private static final int MAX_SEQUENTIAL_TOOL_EXECUTIONS = 100;

    private final ServiceOutputParser serviceOutputParser = new ServiceOutputParser();
    private final Collection<TokenStreamAdapter> tokenStreamAdapters = loadFactories(TokenStreamAdapter.class);

    DefaultAiServices(AiServiceContext context) {
        super(context);
    }

    private static String toString(Object arg) {
        if (arg.getClass().isArray()) {
            return arrayToString(arg);
        } else if (arg.getClass().isAnnotationPresent(StructuredPrompt.class)) {
            return StructuredPromptProcessor.toPrompt(arg).text();
        } else {
            return arg.toString();
        }
    }

    private static String arrayToString(Object arg) {
        StringBuilder sb = new StringBuilder("[");
        int length = Array.getLength(arg);
        for (int i = 0; i < length; i++) {
            sb.append(toString(Array.get(arg, i)));
            if (i < length - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }

    public T build() {

        performBasicValidation();

        for (Method method : context.aiServiceClass.getMethods()) {
            if (method.isAnnotationPresent(Moderate.class) && context.moderationModel == null) {
                throw illegalConfiguration("The @Moderate annotation is present, but the moderationModel is not set up. " +
                        "Please ensure a valid moderationModel is configured before using the @Moderate annotation.");
            }
            if (method.getReturnType() == Result.class ||
                    method.getReturnType() == List.class ||
                    method.getReturnType() == Set.class) {
                TypeUtils.validateReturnTypesAreProperlyParametrized(method.getName(), method.getGenericReturnType());
            }
        }

        Object proxyInstance = Proxy.newProxyInstance(
                context.aiServiceClass.getClassLoader(),
                new Class<?>[]{context.aiServiceClass},
                new InvocationHandler() {

                    private final ExecutorService executor = Executors.newCachedThreadPool();

                    @Override
                    public Object invoke(Object proxy, Method method, Object[] args) throws Exception {

                        if (method.getDeclaringClass() == Object.class) {
                            // methods like equals(), hashCode() and toString() should not be handled by this proxy
                            return method.invoke(this, args);
                        }

                        AiServicesMethodParameter aiServicesMethodParameter = AiServicesMethodParameter.from(method, args);

                        aiServicesMethodParameter.validate();

                        Object memoryId = aiServicesMethodParameter.findMemoryIdAnnotatedParameter().flatMap(AiServicesParameter::memoryIdObject).orElse(DEFAULT);

                        AiServicesInputMessageResolver aiServicesInputMessageResolver = AiServicesInputMessageResolver.from(aiServicesMethodParameter, aiServicesParameter -> DefaultAiServices.toString(aiServicesParameter.object()));
                        AiServicesInputMessageProcessor aiServicesInputMessageProcessor = AiServicesInputMessageProcessor.from(aiServicesInputMessageResolver, DEFAULT, context);

                        Optional<SystemMessage> systemMessage = aiServicesInputMessageProcessor.systemMessage();
                        UserMessage userMessage = aiServicesInputMessageProcessor.userMessage();

                        List<dev.langchain4j.rag.content.Content> augmentationResultContent = aiServicesInputMessageProcessor.augmentationResult(userMessage).map(AugmentationResult::contents).orElse(null);

                        userMessage = aiServicesInputMessageProcessor.userMessageAugmented(userMessage);

                        userMessage = aiServicesInputMessageProcessor.userMessageOutputFormatInstructed(userMessage);

                        aiServicesInputMessageProcessor.saveChatMemory(systemMessage, userMessage);

                        List<ChatMessage> messages = aiServicesInputMessageProcessor.messages(systemMessage, userMessage);

                        Future<Moderation> moderationFuture = triggerModerationIfNeeded(method, messages);

                        // TODO continue refactoring from here




                        List<ToolSpecification> toolSpecifications = context.toolSpecifications;
                        Map<String, ToolExecutor> toolExecutors = context.toolExecutors;

                        if (context.toolProvider != null) {
                            toolSpecifications = new ArrayList<>();
                            toolExecutors = new HashMap<>();
                            ToolProviderRequest toolProviderRequest = new ToolProviderRequest(memoryId, userMessage);
                            ToolProviderResult toolProviderResult = context.toolProvider.provideTools(toolProviderRequest);
                            if (toolProviderResult != null) {
                                Map<ToolSpecification, ToolExecutor> tools = toolProviderResult.tools();
                                for (ToolSpecification toolSpecification : tools.keySet()) {
                                    toolSpecifications.add(toolSpecification);
                                    toolExecutors.put(toolSpecification.name(), tools.get(toolSpecification));
                                }
                            }
                        }






                        Type returnType = aiServicesInputMessageProcessor.returnType();
                        if (aiServicesInputMessageProcessor.streaming()) {
                            TokenStream tokenStream = new AiServiceTokenStream(
                                    messages,
                                    toolSpecifications,
                                    toolExecutors,
                                    augmentationResultContent,
                                    context,
                                    memoryId
                            );
                            // TODO moderation
                            if (returnType == TokenStream.class) {
                                return tokenStream;
                            } else {
                                return adapt(tokenStream, returnType);
                            }
                        }






                        Response<AiMessage> response;
                        Optional<JsonSchema> jsonSchema = aiServicesInputMessageProcessor.jsonSchema();
                        if (aiServicesInputMessageProcessor.supportsJsonSchema().isPresent() && jsonSchema.isPresent()) {
                            ChatRequest chatRequest = ChatRequest.builder()
                                    .messages(messages)
                                    .toolSpecifications(toolSpecifications)
                                    .responseFormat(ResponseFormat.builder()
                                            .type(JSON)
                                            .jsonSchema(jsonSchema.get())
                                            .build())
                                    .build();

                            ChatResponse chatResponse = context.chatModel.chat(chatRequest);

                            response = new Response<>(
                                    chatResponse.aiMessage(),
                                    chatResponse.tokenUsage(),
                                    chatResponse.finishReason()
                            );
                        } else {
                            // TODO migrate to new API
                            response = toolSpecifications == null
                                    ? context.chatModel.generate(messages)
                                    : context.chatModel.generate(messages, toolSpecifications);
                        }

                        TokenUsage tokenUsageAccumulator = response.tokenUsage();

                        verifyModerationIfNeeded(moderationFuture);

                        int executionsLeft = MAX_SEQUENTIAL_TOOL_EXECUTIONS;
                        List<ToolExecution> toolExecutions = new ArrayList<>();
                        while (true) {

                            if (executionsLeft-- == 0) {
                                throw runtime("Something is wrong, exceeded %s sequential tool executions",
                                        MAX_SEQUENTIAL_TOOL_EXECUTIONS);
                            }

                            AiMessage aiMessage = response.content();

                            if (context.hasChatMemory()) {
                                context.chatMemory(memoryId).add(aiMessage);
                            } else {
                                messages = new ArrayList<>(messages);
                                messages.add(aiMessage);
                            }

                            if (!aiMessage.hasToolExecutionRequests()) {
                                break;
                            }

                            for (ToolExecutionRequest toolExecutionRequest : aiMessage.toolExecutionRequests()) {
                                ToolExecutor toolExecutor = toolExecutors.get(toolExecutionRequest.name());
                                String toolExecutionResult = toolExecutor.execute(toolExecutionRequest, memoryId);
                                toolExecutions.add(ToolExecution.builder()
                                        .request(toolExecutionRequest)
                                        .result(toolExecutionResult)
                                        .build());
                                ToolExecutionResultMessage toolExecutionResultMessage = ToolExecutionResultMessage.from(
                                        toolExecutionRequest,
                                        toolExecutionResult
                                );
                                if (context.hasChatMemory()) {
                                    context.chatMemory(memoryId).add(toolExecutionResultMessage);
                                } else {
                                    messages.add(toolExecutionResultMessage);
                                }
                            }

                            if (context.hasChatMemory()) {
                                messages = context.chatMemory(memoryId).messages();
                            }

                            response = context.chatModel.generate(messages, toolSpecifications);
                            tokenUsageAccumulator = TokenUsage.sum(tokenUsageAccumulator, response.tokenUsage());
                        }

                        response = Response.from(response.content(), tokenUsageAccumulator, response.finishReason());

                        Object parsedResponse = serviceOutputParser.parse(response, returnType);
                        if (typeHasRawClass(returnType, Result.class)) {
                            return Result.builder()
                                    .content(parsedResponse)
                                    .tokenUsage(tokenUsageAccumulator)
                                    .sources(augmentationResultContent)
                                    .finishReason(response.finishReason())
                                    .toolExecutions(toolExecutions)
                                    .build();
                        } else {
                            return parsedResponse;
                        }
                    }

                    private Object adapt(TokenStream tokenStream, Type returnType) {
                        for (TokenStreamAdapter tokenStreamAdapter : tokenStreamAdapters) {
                            if (tokenStreamAdapter.canAdaptTokenStreamTo(returnType)) {
                                return tokenStreamAdapter.adapt(tokenStream);
                            }
                        }
                        throw new IllegalStateException("Can't find suitable TokenStreamAdapter");
                    }

                    private Future<Moderation> triggerModerationIfNeeded(Method method, List<ChatMessage> messages) {
                        if (method.isAnnotationPresent(Moderate.class)) {
                            return executor.submit(() -> {
                                List<ChatMessage> messagesToModerate = removeToolMessages(messages);
                                return context.moderationModel.moderate(messagesToModerate).content();
                            });
                        }
                        return null;
                    }
                });

        return (T) proxyInstance;
    }
}
