console.log("Running Your Extension!!!!")
chrome.browserAction.onClicked.addListener(buttonClicked);

function buttonClicked(tab) {
    let msg = {
        txt: "hello"
    }
    chrome.tabs.sendMessage(tab.id, msg)
}