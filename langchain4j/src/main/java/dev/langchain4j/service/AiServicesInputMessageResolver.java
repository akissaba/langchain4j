package dev.langchain4j.service;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Scanner;
import java.util.function.Function;
import java.util.stream.Stream;

import static dev.langchain4j.exception.IllegalConfigurationException.illegalConfiguration;
import static java.lang.String.join;
import static java.lang.System.lineSeparator;
import static java.util.Optional.of;
import static java.util.Optional.ofNullable;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toMap;

public class AiServicesInputMessageResolver {
    private AiServicesMethodParameter aiServicesMethodParameter;
    private Function<AiServicesParameter, String> converter;

    public AiServicesInputMessageResolver() {
        super();
    }

    public static AiServicesInputMessageResolver from(AiServicesMethodParameter aiServicesMethodParameter, Function<AiServicesParameter, String> converter) {
        return new AiServicesInputMessageResolver()
                .aiServicesMethodParameter(aiServicesMethodParameter)
                .converter(converter);
    }

    protected String template(String type, String resource, String delimiter, String[] value) {
        return ofNullable(resource)
                .filter(r -> !r.isBlank())
                .map(r -> {
                    var method = aiServicesMethodParameter().aiServicesMethod().method();
                    var clazz = method.getDeclaringClass();
                    return ofNullable(clazz.getResourceAsStream(r))
                            .or(() -> ofNullable(clazz.getResourceAsStream("/%s".formatted(r))))
                            .orElseThrow(() -> illegalConfiguration("@%sMessage's resource '%s' not found", type, r));
                })
                .flatMap(inputStream -> {
                    try (var scanner = new Scanner(inputStream).useDelimiter("\\A")) {
                        return of(scanner).filter(Scanner::hasNext).map(Scanner::next);
                    }
                })
                .or(() -> of(join(delimiter, value)).filter(v -> !v.isBlank()))
                .orElseThrow(() -> illegalConfiguration("@%sMessage's template cannot be empty", type));
    }

    protected Optional<String> itVariable(String template) {
        var placeholder = "{{it}}";
        return ofNullable(template).filter(t -> t.contains(placeholder))
                .map(t -> aiServicesMethodParameter().findValidMessageParameter()
                        .map(converter())
                        .orElseThrow(() -> illegalConfiguration("Error: cannot find the value of the prompt template variable \"%s\".", placeholder))
                );
    }

    public Map<String, String> variables(String template) {
        var variables = aiServicesMethodParameter().aiServicesParameters().stream()
                .collect(toMap(
                        AiServicesParameter::variableName,
                        converter(),
                        (existingValue, newValue) -> newValue,
                        LinkedHashMap::new
                ));
        var itVariable = "it";
        itVariable(template).filter(it -> !variables.containsKey(itVariable))
                .ifPresent(it -> variables.put(itVariable, it));
        return variables;
    }

    public Optional<String> systemMessageTemplate() {
        return aiServicesMethodParameter()
                .findSystemMessageAnnotatedMethods()
                .flatMap(AiServicesMethod::systemMessageAnnotationMethod)
                .map(systemMessage -> template("System", systemMessage.fromResource(), systemMessage.delimiter(), systemMessage.value()));
    }

    public Optional<String> userMessageTemplate() {
        var aiServicesMethodParameter = aiServicesMethodParameter();
        var templateFromMethodAnnotation = aiServicesMethodParameter
                .findUserMessageAnnotatedMethod()
                .flatMap(AiServicesMethod::userMessageAnnotationMethod)
                .map(userMessage -> template("User", userMessage.fromResource(), userMessage.delimiter(), userMessage.value()));
        var templateFromNotContentParameterAnnotation = aiServicesMethodParameter
                .findNotContentUserMessageAnnotatedParameter()
                .flatMap(AiServicesParameter::notContentUserMessageAnnotationParameter)
                .map(userMessage -> template("User", userMessage.fromResource(), userMessage.delimiter(), userMessage.value()));
        var templateFromParameter = aiServicesMethodParameter
                .findValidMessageParameter().map(converter());
        return Stream.of(templateFromMethodAnnotation, templateFromNotContentParameterAnnotation, templateFromParameter)
                .flatMap(Optional::stream)
                .collect(joining(lineSeparator()))
                .transform(Optional::of)
                .filter(template -> !template.isBlank());
    }

    public AiServicesMethodParameter aiServicesMethodParameter() {
        return aiServicesMethodParameter;
    }

    public AiServicesInputMessageResolver aiServicesMethodParameter(AiServicesMethodParameter aiServicesMethodParameter) {
        this.aiServicesMethodParameter = aiServicesMethodParameter;
        return this;
    }

    public Function<AiServicesParameter, String> converter() {
        return converter;
    }

    public AiServicesInputMessageResolver converter(Function<AiServicesParameter, String> converter) {
        this.converter = converter;
        return this;
    }
}
