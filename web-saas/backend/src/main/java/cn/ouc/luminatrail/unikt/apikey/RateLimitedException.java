package cn.ouc.luminatrail.unikt.apikey;

/** 开放 API 速率超限（HTTP 429）。 */
public class RateLimitedException extends RuntimeException {

    public RateLimitedException(String message) {
        super(message);
    }
}
