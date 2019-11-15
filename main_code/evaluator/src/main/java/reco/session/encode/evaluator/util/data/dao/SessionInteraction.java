package reco.session.encode.evaluator.util.data.dao;

/**
 * Represents one interaction
 */
public class SessionInteraction {

    private String userId;
    private String sessionId;
    private String itemId;
    private String type;
    
    public String getUserId() {
        return userId;
    }
    public void setUserId(String userId) {
        this.userId = userId;
    }
    public String getSessionId() {
        return sessionId;
    }
    public void setSessionId(String sessionId) {
        this.sessionId = sessionId;
    }
    public String getItemId() {
        return itemId;
    }
    public void setItemId(String itemId) {
        this.itemId = itemId;
    }
    public String getType() {
        return type;
    }
    public void setType(String type) {
        this.type = type;
    }
 
    
}
