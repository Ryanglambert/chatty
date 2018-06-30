console.log("Chrome Extension Go")

// Changing content on page


// listener for stuff that background.js is sending us
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
    console.log(request.txt)
    if (request.txt === "hello") {
        // new stuff
        var text = "";
        if (window.getSelection) {
            text = window.getSelection().toString();
            console.log(text)
        } else if (document.selection && document.selection.type != "Control") {
            text = document.selection.createRange().text;
            console.log(text)
        }
        // end new stuff
        let paragraphs = document.getElementsByTagName('p')
        for (elt of paragraphs) {
            elt.style['background-color'] = '#FF00FF'
        }
    }
});
