import './index.css';
class Chatbox {
  constructor() {
      this.args = {
          openButton: document.querySelector('.chatbox__button'),
          chatBox: document.querySelector('.chatbox__support'),
          sendButton: document.querySelector('.send__button')
      }

      this.state = false;
      this.messages = [];
  }

  display() {
      const {openButton, chatBox, sendButton} = this.args;

      openButton.addEventListener('click', () => this.toggleState(chatBox))

      sendButton.addEventListener('click', () => this.onSendButton(chatBox))

      const node = chatBox.querySelector('input');

      node.addEventListener("keyup", ({key}) => {
          if (key === "Enter") {
              this.onSendButton(chatBox)
          }
      })
  }

  toggleState(chatbox) {
      this.state = !this.state;

      // show or hides the box
      if(this.state) {
          chatbox.classList.add('chatbox--active')
          let welcome = { name: "FootBot", message: "Welcome to the FIFA World Cup Chatbot! What's your name?"};
          this.messages.push(welcome)
          this.updateChatText(chatbox)
      } else {
          chatbox.classList.remove('chatbox--active')
      }
  }

  onSendButton(chatbox) {
      var textField = chatbox.querySelector('input');
      let text1 = textField.value
      if (text1 === "") {
          return;
      }
      if (this.messages.length == 0){
          let welcome = { name: "FootBot", message: "Welcome to the FIFA World Cup Chatbot! What's your name?"};
          this.messages.push(welcome);
          return
      }
      let msg1 = { name: "User", message: text1 }
      this.messages.push(msg1);

      fetch('http://127.0.0.1:5000/message', {
          method: 'POST',
          body: JSON.stringify({ prompt: this.messages, message: text1 }),
          mode: 'cors',
          headers: {
            'Content-Type': 'application/json'
          },
        })
        .then(r => r.json())
        .then(r => {
          let msg2 = { name: "FootBot", message: r.answer };
          this.messages.push(msg2);
          // this.updateChatText(chatbox)
          let prompt2 = { name: "FootBot", message: r.nextPrompt };
          this.messages.push(prompt2);
          this.updateChatText(chatbox)
          textField.value = ''

      }).catch((error) => {
          console.error('Error:', error);
          this.updateChatText(chatbox)
          textField.value = ''
        });
  }

  updateChatText(chatbox) {
      var html = '';
      this.messages.slice().reverse().forEach(function(item, index) {
          if (item.name === "FootBot")
          {
              html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>'
          }
          else
          {
              html += '<div class="messages__item messages__item--operator">' + item.message + '</div>'
          }
        });

      const chatmessage = chatbox.querySelector('.chatbox__messages');
      chatmessage.innerHTML = html;
  }
}


const chatbox = new Chatbox();
chatbox.display();