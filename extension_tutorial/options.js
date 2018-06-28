let page = document.getElementById('buttonDiv')
const kButtonColors = ['#3aa757', '#e8453', '#f9bb2d', '#4688f1'];
function constructOptions(kButtonColors) {
    for (let item of kButtonColors) {
        let button = document.createElement('button');
        button.style.backgroundColor = item;
        button.addEventListener('click', function() {
            // storing the color into chrome.storage
            chrome.storage.sync.set({color: item}, function() {
                console.log('color is ' + item);
            })
        });
        page.appendChild(button);
    }
}
constructOptions(kButtonColors);