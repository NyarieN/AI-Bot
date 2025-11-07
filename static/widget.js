// ---------------------------
// Unified Kai Widget
// ---------------------------

// Get business ID and mode
const scriptTag = document.getElementById("kai-widget");
const BUSINESS_ID = scriptTag ? scriptTag.getAttribute("data-business") : "default";
const AI_MODE = scriptTag ? scriptTag.getAttribute("data-ai") === "true" : false;
window.KAI_BUSINESS_ID = BUSINESS_ID;

// Create floating bubble
const bubble = document.createElement("div");
bubble.id = "kai-bubble";
bubble.innerText = "Kai 🤖";
Object.assign(bubble.style, {
  position: "fixed",
  bottom: "20px",
  right: "20px",
  width: "60px",
  height: "60px",
  background: "#333",
  color: "#fff",
  borderRadius: "50%",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  zIndex: 9999
});
document.body.appendChild(bubble);

// Create chat window
const chatWindow = document.createElement("div");
chatWindow.id = "kai-chat-window";
Object.assign(chatWindow.style, {
  position: "fixed",
  bottom: "90px",
  right: "20px",
  width: "300px",
  height: "400px",
  background: "#fff",
  border: "1px solid #ccc",
  borderRadius: "8px",
  display: "none",
  flexDirection: "column",
  zIndex: 9998,
  boxShadow: "0 4px 12px rgba(0,0,0,0.15)"
});
document.body.appendChild(chatWindow);

// Messages container
const messages = document.createElement("div");
messages.id = "messages";
Object.assign(messages.style, {
  flex: "1",
  padding: "10px",
  overflowY: "auto",
  fontFamily: "Arial, sans-serif"
});
chatWindow.appendChild(messages);

// ---------- Welcome message ----------
const welcomeMsg = document.createElement("div");
Object.assign(welcomeMsg.style, { 
    margin: "5px 0",
    textAlign: "left",
    backgroundColor: "#f0f0f0",
    padding: "8px 10px",
    borderRadius: "8px",
    maxWidth: "80%",
    wordWrap: "break-word"
});
welcomeMsg.innerHTML = `Hi, I am Kai, your assistant bot at ${BUSINESS_ID}. How can I help you today?`;
messages.appendChild(welcomeMsg);
scrollToBottom();

// Input area
const inputArea = document.createElement("div");
Object.assign(inputArea.style, { display: "flex", borderTop: "1px solid #ccc" });
chatWindow.appendChild(inputArea);

const userInput = document.createElement("input");
userInput.id = "userInput";
userInput.type = "text";
userInput.placeholder = "Ask me anything...";
Object.assign(userInput.style, { flex: "1", padding: "10px", border: "none" });
inputArea.appendChild(userInput);

const sendButton = document.createElement("button");
sendButton.innerText = "Send";
Object.assign(sendButton.style, { padding: "10px", background: "#333", color: "#fff", border: "none", cursor: "pointer" });
inputArea.appendChild(sendButton);

// Toggle chat window
bubble.addEventListener("click", () => {
  chatWindow.style.display = chatWindow.style.display === "none" ? "flex" : "none";
});

// Send message function
function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Show user message
  const userMsg = document.createElement("div");
  Object.assign(userMsg.style, { margin: "5px 0", textAlign: "right" });
  userMsg.innerHTML = `${text}`;
  messages.appendChild(userMsg);
  userInput.value = "";
  scrollToBottom();

  // Decide which backend route
  const route = AI_MODE ? "/ask" : "/query";

  fetch(route, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      business_id: BUSINESS_ID,
      message: text,
      question: text // used for PDF mode
    })
  })
  .then(res => res.json())
  .then(data => {
    const botMsg = document.createElement("div");
    Object.assign(botMsg.style, { 
        margin: "5px 0",
        textAlign: "left",
        backgroundColor: "#f0f0f0",
        padding: "8px 10px",
        borderRadius: "8px",
        maxWidth: "80%",
        wordWrap: "break-word" 
    });
    botMsg.innerHTML = `${data.reply || data.answer}`;
    messages.appendChild(botMsg);
    scrollToBottom();
  })
  .catch(err => {
    const botMsg = document.createElement("div");
    Object.assign(botMsg.style, { margin: "5px 0", textAlign: "left" });
    botMsg.innerHTML = `<b>Kai:</b> Sorry, something went wrong.`;
    messages.appendChild(botMsg);
    scrollToBottom();
    console.error(err);
  });
}

sendButton.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendMessage();
});

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}
