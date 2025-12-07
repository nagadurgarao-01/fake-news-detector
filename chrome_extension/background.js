chrome.runtime.onInstalled.addListener(() => {
  console.log("Fake News Detector installed.");
  chrome.action.setBadgeText({ text: "ON" });
  chrome.action.setBadgeBackgroundColor({ color: "#4CAF50" });
});
