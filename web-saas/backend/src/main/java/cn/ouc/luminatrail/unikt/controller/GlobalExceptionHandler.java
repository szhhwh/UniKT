package cn.ouc.luminatrail.unikt.controller;

import cn.ouc.luminatrail.unikt.dto.ApiError;
import cn.ouc.luminatrail.unikt.service.InferenceService.InferenceUnavailableException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.client.HttpClientErrorException;
import org.springframework.web.client.RestClientException;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(InferenceUnavailableException.class)
    public ResponseEntity<ApiError> inferenceDown(InferenceUnavailableException e) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(ApiError.of(503, e.getMessage()));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiError> badRequest(MethodArgumentNotValidException e) {
        String detail = e.getBindingResult().getFieldErrors().stream()
                .map(err -> err.getField() + ": " + err.getDefaultMessage())
                .findFirst()
                .orElse("参数校验失败");
        return ResponseEntity.badRequest().body(ApiError.of(400, detail));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> illegalArg(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(ApiError.of(400, e.getMessage()));
    }

    @ExceptionHandler(HttpClientErrorException.class)
    public ResponseEntity<ApiError> upstream4xx(HttpClientErrorException e) {
        String detail = e.getResponseBodyAsString().isBlank()
                ? e.getMessage()
                : e.getResponseBodyAsString();
        return ResponseEntity.status(e.getStatusCode()).body(ApiError.of(e.getStatusCode().value(), detail));
    }

    @ExceptionHandler(RestClientException.class)
    public ResponseEntity<ApiError> upstream(RestClientException e) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(ApiError.of(502, "上游服务错误：" + e.getMessage()));
    }
}
