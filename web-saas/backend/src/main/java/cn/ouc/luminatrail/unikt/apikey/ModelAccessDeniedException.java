package cn.ouc.luminatrail.unikt.apikey;

/** 开放 API 访问了不属于该密钥主人的限定模型（HTTP 404，不泄露存在性）。 */
public class ModelAccessDeniedException extends RuntimeException {

    public ModelAccessDeniedException(String message) {
        super(message);
    }
}
