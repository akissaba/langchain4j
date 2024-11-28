package dev.langchain4j.service;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.TextContent;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.memory.ChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.AugmentationResult;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.output.ServiceOutputParser;
import dev.langchain4j.spi.services.TokenStreamAdapter;

import java.lang.reflect.Type;
import java.util.Collection;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.stream.Stream;

import static dev.langchain4j.model.chat.Capability.RESPONSE_FORMAT_JSON_SCHEMA;
import static dev.langchain4j.service.output.JsonSchemas.jsonSchemaFrom;
import static dev.langchain4j.spi.ServiceHelper.loadFactories;
import static java.lang.System.lineSeparator;
import static java.util.Optional.ofNullable;
import static java.util.function.Function.identity;
import static java.util.stream.Collectors.joining;

public class AiServicesInputMessageProcessor {
    private AiServicesInputMessageResolver aiServicesInputMessageResolver;
    private String defaultMemoryId;
    private AiServiceContext aiServiceContext;
    private ServiceOutputParser serviceOutputParser = new ServiceOutputParser();
    private Collection<TokenStreamAdapter> tokenStreamAdapters = loadFactories(TokenStreamAdapter.class);

    public AiServicesInputMessageProcessor() {
        super();
    }

    public static AiServicesInputMessageProcessor from(AiServicesInputMessageResolver aiServicesInputMessageResolver, String defaultMemoryId, AiServiceContext aiServiceContext) {
        return new AiServicesInputMessageProcessor()
                .aiServicesInputMessageResolver(aiServicesInputMessageResolver)
                .defaultMemoryId(defaultMemoryId)
                .aiServiceContext(aiServiceContext)
                .serviceOutputParser(new ServiceOutputParser())
                .tokenStreamAdapters(loadFactories(TokenStreamAdapter.class));
    }

    protected Optional<String> memoryId() {
        var aiServicesInputMessageResolver = aiServicesInputMessageResolver();
        return aiServicesInputMessageResolver
                .aiServicesMethodParameter()
                .findMemoryIdAnnotatedParameter()
                .map(aiServicesInputMessageResolver.converter())
                .or(() -> ofNullable(defaultMemoryId()));
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
                                return PromptTemplate.from(userMessageTemplate).apply(templateValues);
                            })
                            .map(Prompt::text)
                            .map(TextContent::from).stream();
                    var contents = aiServicesMethodParameter.findValidContentInstances().stream()
                            .flatMap(aiServicesParameter -> aiServicesParameter.validContentInstance().stream());
                    var userMessageContents = Stream.of(textContent, contents).flatMap(identity()).toList();
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
        return ofNullable(aiServiceContext())
                .map(aiServiceContext -> {
                    var name = userMessage.name();
                    var textContent = textContent(userMessage);
                    var memoryId = memoryId();
                    var chatMemory = memoryId
                            .filter(mi -> aiServiceContext.hasChatMemory())
                            .map(mi -> aiServiceContext.chatMemory(mi).messages());
                    var userMessageText = UserMessage.from(name, textContent.stream().toList());
                    var metadata = Metadata.from(userMessageText, memoryId.orElse(null), chatMemory.orElse(null));
                    var augmentationRequest = new AugmentationRequest(userMessageText, metadata);
                    return aiServiceContext.retrievalAugmentor.augment(augmentationRequest);
                })
                .map(augmented -> ofNullable(augmented.chatMessage())
                        .filter(UserMessage.class::isInstance)
                        .map(UserMessage.class::cast)
                        .map(um -> UserMessage.from(userMessage.name(), Stream.of(um.contents(), mediaContents(userMessage)).flatMap(List::stream).toList()))
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
        var returnType = returnType();
        return supportsJsonSchema()
                .flatMap(chatModel -> jsonSchemaFrom(returnType));
    }

    // TODO append after storing in the memory?
    public UserMessage userMessageOutputFormatInstructed(UserMessage userMessage) {
        if (!streaming()) {
            var returnType = returnType();
            var supportsJsonSchema = supportsJsonSchema();
            var jsonSchema = jsonSchema();
            if (supportsJsonSchema.isEmpty() || jsonSchema.isEmpty()) {
                var name = userMessage.name();
                var contents = userMessage.contents();
                // TODO give user ability to provide custom OutputParser
                var instructions = List.<Content>of(TextContent.from(serviceOutputParser().outputFormatInstructions(returnType)));
                var userMessageContents = Stream.of(contents, instructions).flatMap(List::stream).toList();
                userMessage = UserMessage.from(name, userMessageContents);
            }
        }
        return userMessage;
    }

    public Optional<ChatMemory> chatMemory() {
        return ofNullable(aiServiceContext())
                .filter(AiServiceContext::hasChatMemory)
                .map(aiServiceContext -> aiServiceContext.chatMemory(memoryId().orElse(null)));
    }

    public void saveChatMemory(Optional<SystemMessage> systemMessage, UserMessage userMessage) {
        chatMemory().ifPresent(chatMemory ->
                Stream.of(systemMessage.stream(), Stream.of(userMessage)).flatMap(identity())
                        .forEach(chatMemory::add)
        );
    }

    public List<ChatMessage> messages(Optional<SystemMessage> systemMessage, UserMessage userMessage) {
        return chatMemory().map(ChatMemory::messages)
                .orElseGet(() -> Stream.of(systemMessage.stream(), Stream.of(userMessage)).flatMap(identity()).toList());
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

    public AiServiceContext aiServiceContext() {
        return aiServiceContext;
    }

    public AiServicesInputMessageProcessor aiServiceContext(AiServiceContext aiServiceContext) {
        this.aiServiceContext = aiServiceContext;
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
}
