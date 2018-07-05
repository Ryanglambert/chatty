var options = [
    set0 = ['Option 1', 'Option 2'],
    set1 = ['First Option', 'Second Option', 'Third Option']
];

function makeUL(array) {
    var list = document.createElement('ul');

    for(var i = 0; i < array.length; i++) {
        var item = document.createElement('li');
        item.appendChild(document.createTextNode(array[i]));
        list.appendChild(item);
    }

    return list
}

function makeList() {
    var list = makeUL(options[0]);
    ul = document.getElementById('analysisComments');
    console.log(list);
    ul.appendChild(list);
}

function requestAnalysis () {
    var request = new XMLHttpRequest();
    request.open('get', 'http://127.0.0.1:5000/hello', false);
    request.send();
    if (request.readyState == 4 && request.status == 200) {
        var res = JSON.parse(request.responseText);
        return res
    };
};

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
    console.log(text)
    console.log(text.hello)
}

function requestAnalysisAsync(callback) {
    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (request.readyState == 4 && request.status == 200) {
            var text = JSON.parse(request.responseText)
            callback(text)
        }
    }
    request.open('get', 'http://127.0.0.1:5000/hello', true);
    request.send();
}
