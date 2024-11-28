package dev.langchain4j.service;

import java.lang.reflect.Method;
import java.util.List;
import java.util.Optional;
import java.util.function.Predicate;

import static dev.langchain4j.exception.IllegalConfigurationException.illegalConfiguration;
import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static java.util.Optional.ofNullable;

public class AiServicesMethodParameter {
    private AiServicesMethod aiServicesMethod;
    private List<AiServicesParameter> aiServicesParameters;

    public AiServicesMethodParameter() {
        super();
    }

    public static AiServicesMethodParameter from(Method method, Object[] objects) {
        var parameters = method.getParameters();
        var aiServicesMethod = AiServicesMethod.from(method);
        var aiServicesParameters = AiServicesParameter.from(parameters, objects);
        return new AiServicesMethodParameter()
                .aiServicesMethod(aiServicesMethod)
                .aiServicesParameters(aiServicesParameters);
    }

    public Optional<AiServicesMethod> findSystemMessageAnnotatedMethods() {
        return ofNullable(aiServicesMethod()).filter(aiServicesMethod ->
                aiServicesMethod.systemMessageAnnotatedMethod().isPresent()
        );
    }

    public Optional<AiServicesParameter> findNotContentUserMessageAnnotatedParameter() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.notContentUserMessageAnnotatedParameter().isPresent()
        ).stream().findFirst();
    }

    public Optional<AiServicesMethod> findUserMessageAnnotatedMethod() {
        return ofNullable(aiServicesMethod()).filter(aiServicesMethod ->
                aiServicesMethod.userMessageAnnotatedMethod().isPresent()
        );
    }

    public List<AiServicesParameter> findSystemMessageInstances() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.systemMessageInstance().isPresent()
        );
    }

    public Optional<AiServicesParameter> findSystemMessageInstance() {
        return findSystemMessageInstances().stream().findFirst();
    }

    public List<AiServicesParameter> findUserMessageInstances() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.userMessageInstance().isPresent()
        );
    }

    public Optional<AiServicesParameter> findUserMessageInstance() {
        return findUserMessageInstances().stream().findFirst();
    }

    public List<AiServicesParameter> findValidContentInstances() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.validContentInstance().isPresent()
        );
    }

    public Optional<AiServicesParameter> findMemoryIdAnnotatedParameter() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.memoryIdAnnotatedParameter().isPresent()
        ).stream().findFirst();
    }

    public Optional<AiServicesParameter> findUserNameAnnotatedParameter() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.userNameAnnotatedParameter().isPresent()
        ).stream().findFirst();
    }

    public Optional<AiServicesParameter> findValidMessageParameter() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.validMessageObject().isPresent()
        ).stream().findFirst();
    }

    public List<AiServicesParameter> findValidUserMessageParameters() {
        return findBy(aiServicesParameter ->
                aiServicesParameter.userMessageInstance().isPresent()
                        || aiServicesParameter.userMessageAnnotatedParameter().isPresent()
                        || aiServicesParameter.validContentInstance().isPresent()
                        || aiServicesParameter.validMessageObject().isPresent()
        );
    }

    public List<AiServicesParameter> findBy(Predicate<AiServicesParameter> predicate) {
        return aiServicesParameters().stream()
                .filter(predicate)
                .toList();
    }

    public void validate() {
        var method = aiServicesMethod().method().getName();

        if (findSystemMessageInstances().size() > 1) {
            throw illegalConfiguration(
                    "The method '%s' has multiple SystemMessage instances parameters. Please use only one.",
                    method
            );
        }

        if (findUserMessageInstances().size() > 1) {
            throw illegalConfiguration(
                    "The method '%s' has multiple UserMessage instances parameters. Please use only one.",
                    method
            );
        }

        if (findUserMessageAnnotatedMethod().isPresent() && findNotContentUserMessageAnnotatedParameter().isPresent()) {
            throw illegalConfiguration(
                    "Error: The method '%s' has multiple @UserMessage annotations. Please use only one.",
                    method
            );
        }

        if (findValidUserMessageParameters().isEmpty()) {
            throw illegalConfiguration(
                    "Error: The method '%s' does not have a user message defined.",
                    method
            );
        }

        var withMemoryIdParameter = findMemoryIdAnnotatedParameter();
        if (withMemoryIdParameter.isPresent() && withMemoryIdParameter.flatMap(AiServicesParameter::memoryIdObject).isEmpty()) {
            var parameter = withMemoryIdParameter.get().parameter();
            throw illegalArgument(
                    "The value of parameter '%s' annotated with @MemoryId in method '%s' must not be null",
                    parameter.getName(),
                    method
            );
        }
    }

    public AiServicesMethod aiServicesMethod() {
        return aiServicesMethod;
    }

    public AiServicesMethodParameter aiServicesMethod(AiServicesMethod aiServicesMethod) {
        this.aiServicesMethod = aiServicesMethod;
        return this;
    }

    public List<AiServicesParameter> aiServicesParameters() {
        return aiServicesParameters;
    }

    public AiServicesMethodParameter aiServicesParameters(List<AiServicesParameter> aiServicesParameters) {
        this.aiServicesParameters = aiServicesParameters;
        return this;
    }
}
