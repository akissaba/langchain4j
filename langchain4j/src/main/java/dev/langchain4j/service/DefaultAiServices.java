package dev.langchain4j.service;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.moderation.Moderation;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.spi.services.TokenStreamAdapter;

import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static dev.langchain4j.exception.IllegalConfigurationException.illegalConfiguration;
import static dev.langchain4j.internal.Exceptions.runtime;
import static dev.langchain4j.service.TypeUtils.typeHasRawClass;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;

class DefaultAiServices<T> extends AiServices<T> {

    private static final int MAX_SEQUENTIAL_TOOL_EXECUTIONS = 100;

    private final ServiceOutputParser serviceOutputParser = new ServiceOutputParser();
    private final Collection<TokenStreamAdapter> tokenStreamAdapters = loadFactories(TokenStreamAdapter.class);

    DefaultAiServices(AiServiceContext context) {
        super(context);
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

                        AiServicesInputMessageResolver aiServicesInputMessageResolver = AiServicesInputMessageResolver.from(aiServicesMethodParameter);
                        AiServicesInputMessageProcessor aiServicesInputMessageProcessor = AiServicesInputMessageProcessor.from(serviceOutputParser, tokenStreamAdapters, aiServicesInputMessageResolver, DEFAULT, context, executor);

                        UserMessage userMessage = aiServicesInputMessageProcessor.userMessage();

                        List<dev.langchain4j.rag.content.Content> augmentationResultContent = aiServicesInputMessageProcessor.augmentationResult(userMessage)
                                .map(AugmentationResult::contents).orElse(null);

                        userMessage = aiServicesInputMessageProcessor.userMessageAugmented(userMessage);

                        userMessage = aiServicesInputMessageProcessor.userMessageOutputFormatInstructed(userMessage);

                        aiServicesInputMessageProcessor.saveChatMemory(userMessage);

                        Optional<Object> tokenStream = aiServicesInputMessageProcessor.tokenStream(userMessage);

                        if (tokenStream.isPresent()) {
                            return tokenStream.get();
                        }

                        List<ChatMessage> messages = aiServicesInputMessageProcessor.messages(userMessage);

                        Optional<Future<Moderation>> moderationFuture = aiServicesInputMessageProcessor.triggerModerationIfNeeded(userMessage);

                        AiServicesOutputMessageProcessor aiServicesOutputMessageProcessor = AiServicesOutputMessageProcessor.from(aiServicesInputMessageProcessor);

                        Response<AiMessage> response = aiServicesOutputMessageProcessor.response(userMessage);

                        List<ToolSpecification> toolSpecifications = aiServicesInputMessageProcessor.toolSpecifications(userMessage);

                        Map<String, ToolExecutor> toolExecutors = aiServicesInputMessageProcessor.toolExecutors(userMessage);

                        // TODO continue refactoring from here


                        Type returnType = aiServicesInputMessageProcessor.returnType();


                        TokenUsage tokenUsageAccumulator = response.tokenUsage();

                        aiServicesInputMessageProcessor.verifyModerationIfNeeded(moderationFuture);

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
                });

        return (T) proxyInstance;
    }
}
