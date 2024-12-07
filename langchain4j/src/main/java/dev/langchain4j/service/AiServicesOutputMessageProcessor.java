package dev.langchain4j.service;

import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.output.Response;

import static dev.langchain4j.model.chat.request.ResponseFormatType.JSON;
import static java.util.Optional.ofNullable;

public class AiServicesOutputMessageProcessor {
    private AiServicesInputMessageProcessor aiServicesInputMessageProcessor;

    public AiServicesOutputMessageProcessor() {
        super();
    }

    public static AiServicesOutputMessageProcessor from(AiServicesInputMessageProcessor aiServicesInputMessageProcessor) {
        return new AiServicesOutputMessageProcessor()
                .aiServicesInputMessageProcessor(aiServicesInputMessageProcessor);
    }

    public Response<AiMessage> response(UserMessage userMessage) {
        var aiServicesInputMessageProcessor = aiServicesInputMessageProcessor();
        return aiServicesInputMessageProcessor
                .supportsJsonSchema()
                .flatMap(chatLanguageModel -> aiServicesInputMessageProcessor.jsonSchema())
                .map(jsonSchema -> {
                    var chatRequest = ChatRequest.builder()
                            .messages(aiServicesInputMessageProcessor.messages(userMessage))
                            .toolSpecifications(aiServicesInputMessageProcessor.toolSpecifications(userMessage))
                            .responseFormat(ResponseFormat.builder()
                                    .type(JSON)
                                    .jsonSchema(jsonSchema)
                                    .build()
                            )
                            .build();
                    var chatResponse = aiServicesInputMessageProcessor
                            .aiServiceContext()
                            .chatModel
                            .chat(chatRequest);
                    return new Response<>(
                            chatResponse.aiMessage(),
                            chatResponse.tokenUsage(),
                            chatResponse.finishReason()
                    );
                })
                .or(() -> ofNullable(aiServicesInputMessageProcessor.toolSpecifications(userMessage))
                        .map(toolSpecifications -> aiServicesInputMessageProcessor
                                .aiServiceContext()
                                .chatModel
                                .generate(aiServicesInputMessageProcessor.messages(userMessage), toolSpecifications)
                        )
                )
                .orElseGet(() -> aiServicesInputMessageProcessor
                        .aiServiceContext()
                        .chatModel
                        .generate(aiServicesInputMessageProcessor.messages(userMessage))
                );
    }

    public AiServicesInputMessageProcessor aiServicesInputMessageProcessor() {
        return aiServicesInputMessageProcessor;
    }

    public AiServicesOutputMessageProcessor aiServicesInputMessageProcessor(AiServicesInputMessageProcessor aiServicesInputMessageProcessor) {
        this.aiServicesInputMessageProcessor = aiServicesInputMessageProcessor;
        return this;
    }
}
