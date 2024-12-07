package dev.langchain4j.service;

import static dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA;
import static dev.langchain4j.service.output.JsonSchemas.jsonSchemaFrom;
import static java.lang.System.lineSeparator;
import static java.util.Map.entry;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toMap;
import static java.util.stream.Stream.concat;
import static java.util.stream.Stream.of;

import java.lang.reflect.Type;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Future;
import java.util.stream.Stream;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.moderation.Moderation;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.service.tool.ToolExecutor;
import dev.langchain4j.service.tool.ToolProviderRequest;
import dev.langchain4j.service.tool.ToolProviderResult;
import dev.langchain4j.spi.services.TokenStreamAdapter;

public class AiServicesInputMessageProcessor {
    private AiServicesInputMessageResolver aiServicesInputMessageResolver;
    private String defaultMemoryId;
    private AiServiceContext aiServiceContext;
    private ServiceOutputParser serviceOutputParser;
    private Collection<TokenStreamAdapter> tokenStreamAdapters;
    private ExecutorService executorService;

    public AiServicesInputMessageProcessor() {
        super();
    }

    public static AiServicesInputMessageProcessor from(ServiceOutputParser serviceOutputParser, Collection<TokenStreamAdapter> tokenStreamAdapters, AiServicesInputMessageResolver aiServicesInputMessageResolver, String defaultMemoryId, AiServiceContext aiServiceContext, ExecutorService executorService) {
        return new AiServicesInputMessageProcessor()
                .serviceOutputParser(serviceOutputParser)
                .tokenStreamAdapters(tokenStreamAdapters)
                .aiServicesInputMessageResolver(aiServicesInputMessageResolver)
                .defaultMemoryId(defaultMemoryId)
                .aiServiceContext(aiServiceContext)
                .executorService(executorService);
    }

    protected String memoryId() {
        var aiServicesInputMessageResolver = aiServicesInputMessageResolver();
        return aiServicesInputMessageResolver
                .aiServicesMethodParameter()
                .findMemoryIdAnnotatedParameter()
                .map(aiServicesInputMessageResolver.converter())
                .orElseGet(this::defaultMemoryId);
    }

    protected Optional<String> systemMessageTemplateFromMemoryId() {
        return aiServicesInputMessageResolver()
                .aiServicesMethodParameter()
                .findMemoryIdAnnotatedParameter()
                .flatMap(AiServicesParameter::memoryIdObject)
                .or(() -> ofNullable(defaultMemoryId()))
                .flatMap(aiServiceContext.systemMessageProvider);
    }

    protected Optional<String> systemMessageTemplate() {
        return aiServicesInputMessageResolver()
                .systemMessageTemplate()
                .or(this::systemMessageTemplateFromMemoryId);
    }

    public Optional<SystemMessage> systemMessage() {
        var servicesInputMessageResolver = aiServicesInputMessageResolver();
        return servicesInputMessageResolver
                .aiServicesMethodParameter()
                .findSystemMessageInstance()
                .flatMap(AiServicesParameter::systemMessageInstance)
                .or(() -> systemMessageTemplate()
                        .map(systemMessageTemplate -> {
                            var templateValues = servicesInputMessageResolver.variables(systemMessageTemplate);
                            return PromptTemplate.from(systemMessageTemplate).apply(templateValues);
                        })
                        .map(Prompt::text)
                        .map(SystemMessage::systemMessage)
                );
    }

    public UserMessage userMessage() {
        var aiServicesInputMessageResolver = aiServicesInputMessageResolver();
        var aiServicesMethodParameter = aiServicesInputMessageResolver.aiServicesMethodParameter();
        return aiServicesMethodParameter.findUserMessageInstance()
                .flatMap(AiServicesParameter::userMessageInstance)
                .orElseGet(() -> {
                    var textContent = aiServicesInputMessageResolver.userMessageTemplate()
                            .map(userMessageTemplate -> {
                                var templateValues = aiServicesInputMessageResolver.variables(userMessageTemplate);
                                return PromptTemplate.from(userMessageTemplate)
                                        .apply(templateValues);
                            })
                            .map(Prompt::text)
                            .map(TextContent::from).stream();
                    var contents = aiServicesMethodParameter.findValidContentInstances().stream()
                            .flatMap(aiServicesParameter -> aiServicesParameter.validContentInstance().stream());
                    var userMessageContents = concat(textContent, contents).toList();
                    return aiServicesMethodParameter.findUserNameAnnotatedParameter().map(aiServicesInputMessageResolver.converter())
                            .map(userName -> UserMessage.from(userName, userMessageContents))
                            .orElseGet(() -> UserMessage.from(userMessageContents));
                });
    }

    protected Optional<Content> textContent(UserMessage userMessage) {
        return userMessage
                .contents().stream()
                .filter(TextContent.class::isInstance).map(TextContent.class::cast)
                .map(TextContent::text)
                .collect(joining(lineSeparator()))
                .transform(Optional::of)
                .filter(text -> !text.isBlank())
                .map(TextContent::from);
    }

    protected List<Content> mediaContents(UserMessage userMessage) {
        return userMessage
                .contents().stream()
                .filter(content -> !(content instanceof TextContent))
                .toList();
    }

    public Optional<AugmentationResult> augmentationResult(UserMessage userMessage) {
        var name = userMessage.name();
        return ofNullable(aiServiceContext())
                .map(aiServiceContext -> {
                    var textContent = textContent(userMessage).stream().toList();
                    var memoryId = memoryId();
                    var chatMemory = chatMemory().map(ChatMemory::messages).orElse(null);
                    var userMessageText = UserMessage.from(name, textContent);
                    var metadata = Metadata.from(userMessageText, memoryId, chatMemory);
                    var augmentationRequest = new AugmentationRequest(userMessageText, metadata);
                    return aiServiceContext.retrievalAugmentor.augment(augmentationRequest);
                })
                .map(augmented -> ofNullable(augmented.chatMessage())
                        .filter(UserMessage.class::isInstance)
                        .map(UserMessage.class::cast)
                        .map(UserMessage::contents)
                        .map(contents -> concat(contents.stream(), mediaContents(userMessage).stream()).toList())
                        .map(contents -> UserMessage.from(name, contents))
                        .map(um -> AugmentationResult.builder().chatMessage(um).build())
                        .orElse(augmented)
                );
    }

    public UserMessage userMessageAugmented(UserMessage userMessage) {
        return augmentationResult(userMessage)
                .map(AugmentationResult::chatMessage)
                .filter(UserMessage.class::isInstance)
                .map(UserMessage.class::cast)
                .orElse(userMessage);
    }

    public Type returnType() {
        return aiServicesInputMessageResolver()
                .aiServicesMethodParameter()
                .aiServicesMethod()
                .method()
                .getGenericReturnType();
    }

    public List<TokenStreamAdapter> tokenStreamAdaptersToReturnType() {
        var returnType = returnType();
        return Stream.ofNullable(tokenStreamAdapters()).flatMap(Collection::stream)
                .filter(tokenStreamAdapter -> tokenStreamAdapter.canAdaptTokenStreamTo(returnType))
                .toList();
    }

    public boolean streaming() {
        return Objects.equals(returnType(), TokenStream.class) || !tokenStreamAdaptersToReturnType().isEmpty();
    }

    public Optional<ChatLanguageModel> supportsJsonSchema() {
        return ofNullable(aiServiceContext())
                .map(aiServiceContext -> aiServiceContext.chatModel)
                .filter(chatModel -> chatModel.supportedCapabilities().contains(RESPONSE_FORMAT_JSON_SCHEMA));
    }

    public Optional<JsonSchema> jsonSchema() {
        return supportsJsonSchema()
                .flatMap(chatModel -> jsonSchemaFrom(returnType()));
    }

    // TODO append after storing in the memory?
    public UserMessage userMessageOutputFormatInstructed(UserMessage userMessage) {
        if (!streaming()) {
            var supportsJsonSchema = supportsJsonSchema();
            var jsonSchema = jsonSchema();
            if (supportsJsonSchema.isEmpty() || jsonSchema.isEmpty()) {
                var name = userMessage.name();
                var contents = userMessage.contents().stream();
                // TODO give user ability to provide custom OutputParser
                var outputFormatInstructions = serviceOutputParser().outputFormatInstructions(returnType());
                var instructions = of(TextContent.from(outputFormatInstructions));
                var userMessageContents = concat(contents, instructions).toList();
                userMessage = UserMessage.from(name, userMessageContents);
            }
        }
        return userMessage;
    }

    public Optional<ChatMemory> chatMemory() {
        return ofNullable(aiServiceContext())
                .filter(AiServiceContext::hasChatMemory)
                .map(aiServiceContext -> aiServiceContext.chatMemory(memoryId()));
    }

    public void saveChatMemory(UserMessage userMessage) {
        chatMemory().ifPresent(chatMemory ->
                concat(systemMessage().stream(), of(userMessage))
                        .forEach(chatMemory::add)
        );
    }

    public List<ChatMessage> messages(UserMessage userMessage) {
        return chatMemory().map(ChatMemory::messages)
                .orElseGet(() -> concat(systemMessage().stream(), of(userMessage)).toList());
    }

    public Optional<Map<ToolSpecification, ToolExecutor>> tools(UserMessage userMessage) {
        return ofNullable(aiServiceContext())
                .map(aiServiceContext -> aiServiceContext.toolProvider)
                .map(toolProvider -> {
                    var toolProviderRequest = new ToolProviderRequest(memoryId(), userMessage);
                    return toolProvider.provideTools(toolProviderRequest);
                })
                .map(ToolProviderResult::tools);
    }

    public List<ToolSpecification> toolSpecifications(UserMessage userMessage) {
        return tools(userMessage)
                .map(Map::keySet).map(LinkedList::new)
                .orElseGet(() -> ofNullable(aiServiceContext())
                        .map(aiServiceContext -> aiServiceContext.toolSpecifications).map(LinkedList::new)
                        .orElse(null)
                );
    }

    public Map<String, ToolExecutor> toolExecutors(UserMessage userMessage) {
        return tools(userMessage)
                .map(map -> map.entrySet().stream()
                        .map(entry -> entry(entry.getKey().name(), entry.getValue()))
                        .collect(toMap(Entry::getKey, Entry::getValue))
                )
                .orElseGet(() -> ofNullable(aiServiceContext())
                        .map(aiServiceContext -> aiServiceContext.toolExecutors).map(LinkedHashMap::new)
                        .orElse(null)
                );
    }

    public Optional<AiServiceTokenStream> aiServiceTokenStream(UserMessage userMessage) {
        return Optional.of(streaming()).filter(Boolean::booleanValue)
                .map(b -> new AiServiceTokenStream(
                        messages(userMessage),
                        toolSpecifications(userMessage),
                        toolExecutors(userMessage),
                        augmentationResult(userMessage).map(AugmentationResult::contents).orElse(null),
                        aiServiceContext(),
                        memoryId()
                ));
    }

    public Optional<Object> tokenStream(UserMessage userMessage) {
        return aiServiceTokenStream(userMessage)
                .map(aiServiceTokenStream ->
                        // TODO moderation
                        Objects.equals(returnType(), TokenStream.class)
                                ? aiServiceTokenStream
                                : tokenStreamAdaptersToReturnType().stream().findFirst()
                                .map(tokenStreamAdapter -> tokenStreamAdapter.adapt(aiServiceTokenStream))
                                .orElseThrow(() -> new IllegalStateException("Can't find suitable TokenStreamAdapter"))
                );
    }

    public Optional<Future<Moderation>> triggerModerationIfNeeded(UserMessage userMessage) {
        return aiServicesInputMessageResolver()
                .aiServicesMethodParameter()
                .aiServicesMethod()
                .moderateAnnotationMethod()
                .map(moderate -> executorService()
                        .submit(() -> {
                            var messagesToModerate = messages(userMessage).stream()
                                    .filter(chatMessage -> !(chatMessage instanceof ToolExecutionResultMessage))
                                    .filter(chatMessage -> !(chatMessage instanceof AiMessage aiMessage && aiMessage.hasToolExecutionRequests()))
                                    .toList();
                            return aiServiceContext()
                                    .moderationModel
                                    .moderate(messagesToModerate)
                                    .content();
                        })
                );
    }

    public void verifyModerationIfNeeded(Optional<Future<Moderation>> moderationFuture) {
        moderationFuture.map(mf -> {
                    try {
                        return mf.get();
                    } catch (InterruptedException | ExecutionException e) {
                        throw new RuntimeException(e);
                    }
                })
                .filter(Moderation::flagged)
                .map(Moderation::flaggedText)
                .ifPresent(flaggedText -> {
                    throw new ModerationException("Text \"%s\" violates content policy".formatted(flaggedText));
                });
    }

    public AiServicesInputMessageResolver aiServicesInputMessageResolver() {
        return aiServicesInputMessageResolver;
    }

    public AiServicesInputMessageProcessor aiServicesInputMessageResolver(AiServicesInputMessageResolver aiServicesInputMessageResolver) {
        this.aiServicesInputMessageResolver = aiServicesInputMessageResolver;
        return this;
    }

    public String defaultMemoryId() {
        return defaultMemoryId;
    }

    public AiServicesInputMessageProcessor defaultMemoryId(String defaultMemoryId) {
        this.defaultMemoryId = defaultMemoryId;
        return this;
    }

    public ServiceOutputParser serviceOutputParser() {
        return serviceOutputParser;
    }

    public AiServicesInputMessageProcessor serviceOutputParser(ServiceOutputParser serviceOutputParser) {
        this.serviceOutputParser = serviceOutputParser;
        return this;
    }

    public Collection<TokenStreamAdapter> tokenStreamAdapters() {
        return tokenStreamAdapters;
    }

    public AiServicesInputMessageProcessor tokenStreamAdapters(Collection<TokenStreamAdapter> tokenStreamAdapters) {
        this.tokenStreamAdapters = tokenStreamAdapters;
        return this;
    }

    public AiServiceContext aiServiceContext() {
        return aiServiceContext;
    }

    public AiServicesInputMessageProcessor aiServiceContext(AiServiceContext aiServiceContext) {
        this.aiServiceContext = aiServiceContext;
        return this;
    }

    public ExecutorService executorService() {
        return executorService;
    }

    public AiServicesInputMessageProcessor executorService(ExecutorService executorService) {
        this.executorService = executorService;
        return this;
    }
}
