document.addEventListener('DOMContentLoaded', function () {
    // Create and append the toggle button to the body
    var toggleButton = document.createElement('button');
    //toggleButton.textContent = 'Toggle Chat';
    //document.body.appendChild(toggleButton);

    // Create and append the chat window iframe container to the body
    var iframeContainer = document.createElement('div');
    iframeContainer.id = 'iframe-container';
    iframeContainer.style.position = 'fixed';
    iframeContainer.style.bottom = '0';
    iframeContainer.style.right = '0';
    iframeContainer.style.display = 'block'; // Initial display set to 'block'
    
    // Create and append the chat window iframe to the iframe container
    var chatWindow = document.createElement('iframe');
    chatWindow.src = 'https://chatbotss.pythonanywhere.com';
    chatWindow.width = '400';
    chatWindow.height = '600';
    chatWindow.frameBorder = '0';
    iframeContainer.appendChild(chatWindow);

    document.body.appendChild(iframeContainer);

    // Initial state (chat window visible)
    var isChatVisible = true;

    // Toggle function
    function toggleChat() {
        if (isChatVisible) {
            iframeContainer.style.display = 'none';
        } else {
            iframeContainer.style.display = 'block';
        }
        isChatVisible = !isChatVisible;
    }

    // Attach click event to the toggle button
    toggleButton.addEventListener('click', toggleChat);
});
