function updateAnalysisDisplay(text) {
    var array = text.hello
    var list = document.createElement('ul');
    for(var i = 0; i < array.length; i++) {
        var item = document.createElement('li');
        item.appendChild(document.createTextNode(array[i]))
        list.appendChild(item);
    }
    ul = document.getElementById('analysisComments');
    ul.appendChild(list)
    // console.log(text)
    // console.log(text.hello)
}

function requestAnalysisAsync(callback) {
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4 && request.status == 200) {
            console.log(request.responseText)
            var text = JSON.parse(request.responseText)
            callback(text)
        }
    }
    request.open('get', analyzeCommentsEndpoint, true);
    request.send();
}
