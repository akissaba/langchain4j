package dev.langchain4j.service;

import java.lang.reflect.Method;
import java.util.Optional;

import static java.util.Optional.ofNullable;

public class AiServicesMethod {
    private Method method;

    public AiServicesMethod() {
        super();
    }

    public static AiServicesMethod from(Method method) {
        return new AiServicesMethod()
                .method(method);
    }

    public Optional<Method> systemMessageAnnotatedMethod() {
        return ofNullable(method()).filter(m -> m.isAnnotationPresent(SystemMessage.class));
    }

    public Optional<SystemMessage> systemMessageAnnotationMethod() {
        return systemMessageAnnotatedMethod().map(m -> m.getAnnotation(SystemMessage.class));
    }

    public Optional<Method> userMessageAnnotatedMethod() {
        return ofNullable(method()).filter(m -> m.isAnnotationPresent(UserMessage.class));
    }

    public Optional<UserMessage> userMessageAnnotationMethod() {
        return userMessageAnnotatedMethod().map(m -> m.getAnnotation(UserMessage.class));
    }

    public Optional<Method> moderateAnnotatedMethod() {
        return ofNullable(method()).filter(m -> m.isAnnotationPresent(Moderate.class));
    }

    public Optional<Moderate> moderateAnnotationMethod() {
        return moderateAnnotatedMethod().map(m -> m.getAnnotation(Moderate.class));
    }

    public Method method() {
        return method;
    }

    public AiServicesMethod method(Method method) {
        this.method = method;
        return this;
    }
}
