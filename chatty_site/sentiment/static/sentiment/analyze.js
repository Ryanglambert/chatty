var options = [
    set0 = ['Option 1', 'Option 2'],
    set1 = ['First Option', 'Second Option', 'Third Option']
];

function makeUL(array) {
    var list = document.createElement('ul');

    for(var i = 0; i < array.length; i++) {
        var item = document.createElement('li');
        item.appendChild(document.createTextNode(array[i]))
        list.appendChild(item);
    }

    return list
}

function makeList() {
    var list = makeUL(options[0])

    return list
}