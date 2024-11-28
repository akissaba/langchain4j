package dev.langchain4j.service;

import static java.util.Optional.ofNullable;
import static java.util.stream.IntStream.range;

import java.lang.reflect.Parameter;
import java.util.List;
import java.util.Optional;

import dev.langchain4j.data.message.Content;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;

public class AiServicesParameter {
    private Parameter parameter;
    private Object object;
    private int index;

    public AiServicesParameter() {
        super();
    }

    public static List<AiServicesParameter> from(Parameter[] parameters, Object[] objects) {
        return range(0, parameters.length)
                .mapToObj(i -> from(parameters[i], objects[i], i))
                .toList();
    }

    public static AiServicesParameter from(Parameter parameter, Object object, int index) {
        return new AiServicesParameter()
                .parameter(parameter)
                .object(object)
                .index(index);
    }

    public Optional<SystemMessage> systemMessageInstance() {
        return ofNullable(object()).filter(SystemMessage.class::isInstance).map(SystemMessage.class::cast);
    }

    public Optional<UserMessage> userMessageInstance() {
        return ofNullable(object()).filter(UserMessage.class::isInstance).map(UserMessage.class::cast);
    }

    public Optional<Content> contentInstance() {
        return ofNullable(object()).filter(Content.class::isInstance).map(Content.class::cast);
    }

    public Optional<Parameter> userMessageAnnotatedParameter() {
        return ofNullable(parameter()).filter(p -> p.isAnnotationPresent(dev.langchain4j.service.UserMessage.class));
    }

    public Optional<dev.langchain4j.service.UserMessage> userMessageAnnotationParameter() {
        return userMessageAnnotatedParameter().map(p -> p.getAnnotation(dev.langchain4j.service.UserMessage.class));
    }

    public Optional<Parameter> notContentUserMessageAnnotatedParameter() {
        return userMessageAnnotatedParameter().filter(p -> contentInstance().isEmpty());
    }

    public Optional<dev.langchain4j.service.UserMessage> notContentUserMessageAnnotationParameter() {
        return userMessageAnnotationParameter().filter(p -> contentInstance().isEmpty());
    }

    public Optional<Parameter> vAnnotatedParameter() {
        return ofNullable(parameter()).filter(p -> p.isAnnotationPresent(V.class));
    }

    public Optional<V> vAnnotationParameter() {
        return vAnnotatedParameter().map(p -> p.getAnnotation(V.class));
    }

    public Optional<Parameter> memoryIdAnnotatedParameter() {
        return ofNullable(parameter()).filter(p -> p.isAnnotationPresent(MemoryId.class));
    }

    public Optional<Object> memoryIdObject() {
        return memoryIdAnnotatedParameter().map(p -> object());
    }

    public Optional<Parameter> userNameAnnotatedParameter() {
        return ofNullable(parameter()).filter(p -> p.isAnnotationPresent(UserName.class));
    }

    public Optional<Object> validMessageObject() {
        return ofNullable(object()).filter(o ->
                systemMessageInstance().isEmpty()
                        && userMessageInstance().isEmpty()
                        && contentInstance().isEmpty()
                        && userMessageAnnotatedParameter().isEmpty()
                        && vAnnotatedParameter().isEmpty()
                        && memoryIdAnnotatedParameter().isEmpty()
                        && userNameAnnotatedParameter().isEmpty()
        );
    }

    public Optional<Content> validContentInstance() {
        return userMessageAnnotatedParameter().flatMap(p -> contentInstance());
    }

    public String variableName() {
        return vAnnotationParameter().map(V::value).orElseGet(parameter()::getName);
    }

    public Parameter parameter() {
        return parameter;
    }

    public AiServicesParameter parameter(Parameter parameter) {
        this.parameter = parameter;
        return this;
    }

    public Object object() {
        return object;
    }

    public AiServicesParameter object(Object object) {
        this.object = object;
        return this;
    }

    public int index() {
        return index;
    }

    public AiServicesParameter index(int index) {
        this.index = index;
        return this;
    }
}
