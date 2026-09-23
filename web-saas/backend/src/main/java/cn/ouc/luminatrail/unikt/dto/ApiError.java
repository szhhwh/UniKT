package cn.ouc.luminatrail.unikt.dto;

import java.time.Instant;

/** 统一错误响应体。 */
public record ApiError(int status, String message, Instant timestamp) {

    public static ApiError of(int status, String message) {
        return new ApiError(status, message, Instant.now());
    }
}
