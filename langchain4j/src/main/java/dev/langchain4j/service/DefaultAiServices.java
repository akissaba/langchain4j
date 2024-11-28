package dev.langchain4j.service;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.input.structured.StructuredPrompt;
import dev.langchain4j.model.input.structured.StructuredPromptProcessor;
import dev.langchain4j.model.moderation.Moderation;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.service.tool.ToolExecution;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.service.tool.ToolProviderResult;
import dev.langchain4j.spi.services.TokenStreamAdapter;

import java.io.InputStream;
import java.lang.reflect.Array;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Parameter;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Scanner;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import static dev.langchain4j.exception.IllegalConfigurationException.illegalConfiguration;
import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static dev.langchain4j.internal.Exceptions.runtime;
import static dev.langchain4j.internal.Utils.isNotNullOrBlank;
import static dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA;
import static dev.langchain4j.model.chat.request.ResponseFormatType.JSON;
import static dev.langchain4j.service.TypeUtils.typeHasRawClass;
import static dev.langchain4j.service.output.JsonSchemas.jsonSchemaFrom;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;

class DefaultAiServices<T> extends AiServices<T> {

    private static final int MAX_SEQUENTIAL_TOOL_EXECUTIONS = 100;

    private final ServiceOutputParser serviceOutputParser = new ServiceOutputParser();
    private final Collection<TokenStreamAdapter> tokenStreamAdapters = loadFactories(TokenStreamAdapter.class);

    DefaultAiServices(AiServiceContext context) {
        super(context);
    }

    static void validateParameters(Method method, Object[] args) {
        Parameter[] parameters = method.getParameters();
        if (parameters.length <= 1) {
            return;
        }

        int userMessageInstancesCount = 0;

        for (int i = 0; i < parameters.length; i++) {
            UserMessage userMessageInstance = args[i] instanceof UserMessage userMessage ? userMessage : null;
            Parameter parameter = parameters[i];
            V v = parameter.getAnnotation(V.class);
            dev.langchain4j.service.UserMessage userMessage = parameter.getAnnotation(dev.langchain4j.service.UserMessage.class);
            MemoryId memoryId = parameter.getAnnotation(MemoryId.class);
            UserName userName = parameter.getAnnotation(UserName.class);
            if (userMessageInstance == null && v == null && userMessage == null && memoryId == null && userName == null) {
                throw illegalConfiguration(
                        "Parameter '%s' of method '%s' should be a UserMessage or annotated with @V or @UserMessage " +
                                "or @UserName or @MemoryId", parameter.getName(), method.getName()
                );
            }

            userMessageInstancesCount += userMessageInstance == null ? 0 : 1;
            if (userMessageInstancesCount > 1) {
                throw illegalConfiguration(
                        "The method '%s' has multiple UserMessage parameters. Please use only one.",
                        method.getName()
                );
            }

            if (userMessageInstancesCount > 0 && (v != null || userMessage != null || userName != null)) {
                throw illegalConfiguration(
                        "The method '%s' has multiple parameters mixed with UserMessage included. Please use only one UserMessage or parameters which is not a UserMessage.",
                        method.getName()
                );
            }
        }
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

                        validateParameters(method, args);

                        Object memoryId = findMemoryId(method, args).orElse(DEFAULT);

                        Optional<SystemMessage> systemMessage = prepareSystemMessage(memoryId, method, args);
                        UserMessage userMessage = prepareUserMessage(method, args);
                        AugmentationResult augmentationResult = null;
                        if (context.retrievalAugmentor != null) {
                            List<ChatMessage> chatMemory = context.hasChatMemory()
                                    ? context.chatMemory(memoryId).messages()
                                    : null;
                            Metadata metadata = Metadata.from(userMessage, memoryId, chatMemory);
                            AugmentationRequest augmentationRequest = new AugmentationRequest(userMessage, metadata);
                            augmentationResult = context.retrievalAugmentor.augment(augmentationRequest);
                            userMessage = (UserMessage) augmentationResult.chatMessage();
                        }

                        // TODO give user ability to provide custom OutputParser
                        Type returnType = method.getGenericReturnType();

                        boolean streaming = returnType == TokenStream.class || canAdaptTokenStreamTo(returnType);

                        boolean supportsJsonSchema = supportsJsonSchema();
                        Optional<JsonSchema> jsonSchema = Optional.empty();
                        if (supportsJsonSchema && !streaming) {
                            jsonSchema = jsonSchemaFrom(returnType);
                        }

                        if ((!supportsJsonSchema || !jsonSchema.isPresent()) && !streaming) {
                            // TODO append after storing in the memory?
                            userMessage = appendOutputFormatInstructions(returnType, userMessage);
                        }

                        if (context.hasChatMemory()) {
                            ChatMemory chatMemory = context.chatMemory(memoryId);
                            systemMessage.ifPresent(chatMemory::add);
                            chatMemory.add(userMessage);
                        }

                        List<ChatMessage> messages;
                        if (context.hasChatMemory()) {
                            messages = context.chatMemory(memoryId).messages();
                        } else {
                            messages = new ArrayList<>();
                            systemMessage.ifPresent(messages::add);
                            messages.add(userMessage);
                        }

                        Future<Moderation> moderationFuture = triggerModerationIfNeeded(method, messages);

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

                        if (streaming) {
                            TokenStream tokenStream = new AiServiceTokenStream(
                                    messages,
                                    toolSpecifications,
                                    toolExecutors,
                                    augmentationResult != null ? augmentationResult.contents() : null,
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
                        if (supportsJsonSchema && jsonSchema.isPresent()) {
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
                                    .sources(augmentationResult == null ? null : augmentationResult.contents())
                                    .finishReason(response.finishReason())
                                    .toolExecutions(toolExecutions)
                                    .build();
                        } else {
                            return parsedResponse;
                        }
                    }

                    private boolean canAdaptTokenStreamTo(Type returnType) {
                        for (TokenStreamAdapter tokenStreamAdapter : tokenStreamAdapters) {
                            if (tokenStreamAdapter.canAdaptTokenStreamTo(returnType)) {
                                return true;
                            }
                        }
                        return false;
                    }

                    private Object adapt(TokenStream tokenStream, Type returnType) {
                        for (TokenStreamAdapter tokenStreamAdapter : tokenStreamAdapters) {
                            if (tokenStreamAdapter.canAdaptTokenStreamTo(returnType)) {
                                return tokenStreamAdapter.adapt(tokenStream);
                            }
                        }
                        throw new IllegalStateException("Can't find suitable TokenStreamAdapter");
                    }

                    private boolean supportsJsonSchema() {
                        return context.chatModel != null
                                && context.chatModel.supportedCapabilities().contains(RESPONSE_FORMAT_JSON_SCHEMA);
                    }

                    private UserMessage appendOutputFormatInstructions(Type returnType, UserMessage userMessage) {
                        String outputFormatInstructions = serviceOutputParser.outputFormatInstructions(returnType);
                        List<Content> contents = new LinkedList<>(userMessage.contents());
                        if (isNotNullOrBlank(outputFormatInstructions)) {
                            contents.add(TextContent.from(outputFormatInstructions));
                        }
                        if (isNotNullOrBlank(userMessage.name())) {
                            userMessage = UserMessage.from(userMessage.name(), contents);
                        } else {
                            userMessage = UserMessage.from(contents);
                        }
                        return userMessage;
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

    private Optional<SystemMessage> prepareSystemMessage(Object memoryId, Method method, Object[] args) {
        return findSystemMessageTemplate(memoryId, method)
                .map(systemMessageTemplate -> PromptTemplate.from(systemMessageTemplate)
                        .apply(findTemplateVariables(systemMessageTemplate, method.getParameters(), args))
                        .toSystemMessage());
    }

    private Optional<String> findSystemMessageTemplate(Object memoryId, Method method) {
        dev.langchain4j.service.SystemMessage annotation = method.getAnnotation(dev.langchain4j.service.SystemMessage.class);
        if (annotation != null) {
            return Optional.of(getTemplate(method, "System", annotation.fromResource(), annotation.value(), annotation.delimiter()));
        }

        return context.systemMessageProvider.apply(memoryId);
    }

    private static Map<String, Object> findTemplateVariables(String template, Parameter[] parameters, Object[] args) {
        Map<String, Object> variables = new HashMap<>();
        for (int i = 0; i < parameters.length; i++) {
            String variableName = getVariableName(parameters[i]);
            Object variableValue = args[i];
            variables.put(variableName, variableValue);
        }

        if (template.contains("{{it}}") && !variables.containsKey("it")) {
            String itValue = getValueOfVariableIt(parameters, args);
            variables.put("it", itValue);
        }

        return variables;
    }

    private static String getVariableName(Parameter parameter) {
        V annotation = parameter.getAnnotation(V.class);
        if (annotation != null) {
            return annotation.value();
        } else {
            return parameter.getName();
        }
    }

    private static String getValueOfVariableIt(Parameter[] parameters, Object[] args) {
        if (parameters.length == 1) {
            Parameter parameter = parameters[0];
            if (!parameter.isAnnotationPresent(MemoryId.class)
                    && !parameter.isAnnotationPresent(dev.langchain4j.service.UserMessage.class)
                    && !parameter.isAnnotationPresent(UserName.class)
                    && (!parameter.isAnnotationPresent(V.class) || isAnnotatedWithIt(parameter))) {
                return toString(args[0]);
            }
        }

        for (int i = 0; i < parameters.length; i++) {
            if (isAnnotatedWithIt(parameters[i])) {
                return toString(args[i]);
            }
        }

        throw illegalConfiguration("Error: cannot find the value of the prompt template variable \"{{it}}\".");
    }

    private static boolean isAnnotatedWithIt(Parameter parameter) {
        V annotation = parameter.getAnnotation(V.class);
        return annotation != null && "it".equals(annotation.value());
    }

    private static UserMessage prepareUserMessage(Method method, Object[] args) {
        Optional<UserMessage> maybeUserMessage = findUserMessage(args);

        if (maybeUserMessage.isPresent()) {
            return maybeUserMessage.get();
        }

        String template = getUserMessageTemplate(method, args);
        Parameter[] parameters = method.getParameters();
        Map<String, Object> variables = findTemplateVariables(template, parameters, args);

        Prompt prompt = PromptTemplate.from(template).apply(variables);

        List<Content> contents = findUserContents(parameters, args);
        contents.add(0, TextContent.from(prompt.text()));

        Optional<String> maybeUserName = findUserName(parameters, args);
        return maybeUserName.map(userName -> UserMessage.from(userName, contents))
                .orElseGet(() -> UserMessage.from(contents));
    }

    private static Optional<UserMessage> findUserMessage(Object[] args) {
        if (args != null) {
            for (Object arg : args) {
                if (arg instanceof UserMessage userMessage) {
                    return Optional.of(userMessage);
                }
            }
        }
        return Optional.empty();
    }

    private static List<Content> findUserContents(Parameter[] parameters, Object[] args) {
        List<Content> contents = new LinkedList<>();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(dev.langchain4j.service.UserMessage.class) && args[i] instanceof Content content) {
                contents.add(content);
            }
        }
        return contents;
    }

    private static String getUserMessageTemplate(Method method, Object[] args) {

        Parameter[] parameters = method.getParameters();
        Optional<String> templateFromMethodAnnotation = findUserMessageTemplateFromMethodAnnotation(method);
        Optional<String> templateFromParameterAnnotation = findUserMessageTemplateFromAnnotatedParameter(parameters, args);

        if (templateFromMethodAnnotation.isPresent() && templateFromParameterAnnotation.isPresent()) {
            throw illegalConfiguration(
                    "Error: The method '%s' has multiple @UserMessage annotations. Please use only one.",
                    method.getName()
            );
        }

        if (templateFromMethodAnnotation.isPresent()) {
            return templateFromMethodAnnotation.get();
        }
        if (templateFromParameterAnnotation.isPresent()) {
            return templateFromParameterAnnotation.get();
        }

        Optional<String> templateFromNotAnnotatedParameter = findUserMessageTemplateFromNotAnnotatedParameter(parameters, args);
        if (templateFromNotAnnotatedParameter.isPresent()) {
            return templateFromNotAnnotatedParameter.get();
        }

        throw illegalConfiguration("Error: The method '%s' does not have a user message defined.", method.getName());
    }

    private static Optional<String> findUserMessageTemplateFromMethodAnnotation(Method method) {
        return Optional.ofNullable(method.getAnnotation(dev.langchain4j.service.UserMessage.class))
                .map(a -> getTemplate(method, "User", a.fromResource(), a.value(), a.delimiter()));
    }

    private static Optional<String> findUserMessageTemplateFromAnnotatedParameter(Parameter[] parameters, Object[] args) {
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(dev.langchain4j.service.UserMessage.class) && !(args[i] instanceof Content)) {
                return Optional.of(toString(args[i]));
            }
        }
        return Optional.empty();
    }

    private static Optional<String> findUserMessageTemplateFromNotAnnotatedParameter(Parameter[] parameters, Object[] args) {
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].getAnnotations().length == 0) {
                return Optional.of(toString(args[i]));
            }
        }
        return Optional.empty();
    }

    private static Optional<String> findUserName(Parameter[] parameters, Object[] args) {
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(UserName.class)) {
                return Optional.of(args[i].toString());
            }
        }
        return Optional.empty();
    }

    private static String getTemplate(Method method, String type, String resource, String[] value, String delimiter) {
        String messageTemplate;
        if (!resource.trim().isEmpty()) {
            messageTemplate = getResourceText(method.getDeclaringClass(), resource);
            if (messageTemplate == null) {
                throw illegalConfiguration("@%sMessage's resource '%s' not found", type, resource);
            }
        } else {
            messageTemplate = String.join(delimiter, value);
        }
        if (messageTemplate.trim().isEmpty()) {
            throw illegalConfiguration("@%sMessage's template cannot be empty", type);
        }
        return messageTemplate;
    }

    private static String getResourceText(Class<?> clazz, String resource) {
        InputStream inputStream = clazz.getResourceAsStream(resource);
        if (inputStream == null) {
            inputStream = clazz.getResourceAsStream("/" + resource);
        }
        return getText(inputStream);
    }

    private static String getText(InputStream inputStream) {
        if (inputStream == null) {
            return null;
        }
        try (Scanner scanner = new Scanner(inputStream);
             Scanner s = scanner.useDelimiter("\\A")) {
            return s.hasNext() ? s.next() : "";
        }
    }

    private static Optional<Object> findMemoryId(Method method, Object[] args) {
        Parameter[] parameters = method.getParameters();
        for (int i = 0; i < parameters.length; i++) {
            if (parameters[i].isAnnotationPresent(MemoryId.class)) {
                Object memoryId = args[i];
                if (memoryId == null) {
                    throw illegalArgument(
                            "The value of parameter '%s' annotated with @MemoryId in method '%s' must not be null",
                            parameters[i].getName(), method.getName()
                    );
                }
                return Optional.of(memoryId);
            }
        }
        return Optional.empty();
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
}
