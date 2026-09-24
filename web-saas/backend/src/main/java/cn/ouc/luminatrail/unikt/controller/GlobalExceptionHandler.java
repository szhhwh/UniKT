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
        String detail = friendlyUpstreamDetail(e.getResponseBodyAsString());
        if (detail == null || detail.isBlank()) {
            detail = e.getMessage();
        }
        return ResponseEntity.status(e.getStatusCode())
                .body(ApiError.of(e.getStatusCode().value(), detail));
    }

    /** 把上游（Python/pydantic）的原始 4xx 体提炼成一句人话，透传原始 JSON 太不友好。 */
    private static String friendlyUpstreamDetail(String body) {
        if (body == null || body.isEmpty()) {
            return null;
        }
        try {
            com.fasterxml.jackson.databind.JsonNode root =
                    new com.fasterxml.jackson.databind.ObjectMapper().readTree(body);
            com.fasterxml.jackson.databind.JsonNode d = root.get("detail");
            if (d != null && d.isArray() && !d.isEmpty()) {
                com.fasterxml.jackson.databind.JsonNode first = d.get(0);
                String msg = first.path("msg").asText(null);
                if (msg == null) {
                    return body;
                }
                String loc = first.path("loc").toString();
                // loc 形如 ["body","responses"]，取字段名拼进提示
                String field = loc.replaceAll("[\\[\\]\"\\s]", "").replace("body,", "");
                return field.isEmpty() ? msg : field + ": " + msg;
            }
            if (d != null && d.isTextual()) {
                return d.asText();
            }
            return body;
        } catch (Exception e) {
            return body;
        }
    }

    @ExceptionHandler(RestClientException.class)
    public ResponseEntity<ApiError> upstream(RestClientException e) {
        return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                .body(ApiError.of(502, "上游服务错误：" + e.getMessage()));
    }
}
