const chat = document.getElementById("chat");
const summaryBox = document.getElementById("summary");
const ragResults = document.getElementById("ragResults");
const userIdInput = document.getElementById("userId");
const messageInput = document.getElementById("message");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");
const queryBtn = document.getElementById("queryBtn");
const clearBtn = document.getElementById("clearBtn");
const calendarDiv = document.getElementById("calendar");
const monthYearSpan = document.getElementById("monthYear");
const prevMonthBtn = document.getElementById("prevMonth");
const nextMonthBtn = document.getElementById("nextMonth");
const activeTitle = document.getElementById("activeTitle");
const activeMeta = document.getElementById("activeMeta");

let activeSessionId = null;
let currentDate = new Date();
let allSessions = [];

function addMessage(text, cls, meta) {
  const div = document.createElement("div");
  div.className = "msg " + cls;
  div.textContent = text;

  if (meta) {
    const span = document.createElement("span");
    span.className = "meta";
    span.textContent = meta;
    div.appendChild(span);
  }

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function clearChat() {
  chat.innerHTML = "";
  summaryBox.style.display = "none";
  summaryBox.textContent = "";
  ragResults.style.display = "none";
  ragResults.textContent = "";
  ragResults.classList.remove("show");
}

function setActiveSession(session) {
  activeSessionId = session ? session.id : null;

  if (session) {
    activeTitle.textContent = session.title || `Day ${session.session_day}`;
    activeMeta.textContent = `${session.session_day} • Updated ${session.updated_at || ""}`;
  } else {
    activeTitle.textContent = "No day selected";
    activeMeta.textContent = "Pick a day from the calendar.";
  }

  renderCalendar();
}

function getDaysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate();
}

function getFirstDayOfMonth(year, month) {
  return new Date(year, month, 1).getDay();
}

function renderCalendar() {
  calendarDiv.innerHTML = "";

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();

  const monthNames = ["January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"];

  monthYearSpan.textContent = `${monthNames[month]} ${year}`;

  const weekdays = document.createElement("div");
  weekdays.className = "calendar-weekdays";
  ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"].forEach(day => {
    const span = document.createElement("div");
    span.className = "weekday";
    span.textContent = day;
    weekdays.appendChild(span);
  });
  calendarDiv.appendChild(weekdays);

  const daysDiv = document.createElement("div");
  daysDiv.className = "calendar-days";

  const firstDay = getFirstDayOfMonth(year, month);
  const daysInMonth = getDaysInMonth(year, month);
  const daysInPrevMonth = getDaysInMonth(year, month - 1);

  for (let i = firstDay - 1; i >= 0; i--) {
    const dayDiv = document.createElement("div");
    dayDiv.className = "day other-month";
    dayDiv.textContent = daysInPrevMonth - i;
    daysDiv.appendChild(dayDiv);
  }

  for (let day = 1; day <= daysInMonth; day++) {
    const dayDiv = document.createElement("div");
    dayDiv.className = "day";

    const dateStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`;
    const session = allSessions.find(s => s.session_day === dateStr);

    if (session) {
      dayDiv.classList.add("has-session");
    }

    if (activeSessionId && session && session.id === activeSessionId) {
      dayDiv.classList.add("active");
    }

    dayDiv.textContent = day;
    dayDiv.addEventListener("click", () => {
      if (session) {
        loadSession(session);
      }
    });

    daysDiv.appendChild(dayDiv);
  }

  const remainingCells = 42 - (firstDay + daysInMonth);
  for (let day = 1; day <= remainingCells; day++) {
    const dayDiv = document.createElement("div");
    dayDiv.className = "day other-month";
    dayDiv.textContent = day;
    daysDiv.appendChild(dayDiv);
  }

  calendarDiv.appendChild(daysDiv);
}

async function loadSessions() {
  const user_id = userIdInput.value.trim();

  if (!user_id) {
    alert("Please enter a user id.");
    return;
  }

  const res = await fetch("/sessions/" + encodeURIComponent(user_id));
  const data = await res.json();
  allSessions = data.sessions || [];
  renderCalendar();
}

async function loadSession(session) {
  setActiveSession(session);
  clearChat();

  const res = await fetch("/sessions/messages/" + encodeURIComponent(session.id));
  const data = await res.json();

  const messages = data.messages || [];
  if (!messages.length) {
    addMessage("No messages in this day yet.", "bot", "System");
    return;
  }

  messages.forEach(row => {
    addMessage(row.user_message, "user", "You");
    addMessage(row.bot_response, "bot", "Bot");
  });

  if (session.summary) {
    summaryBox.textContent = "Summary: " + session.summary;
    summaryBox.style.display = "block";
  }
}

async function sendMessage() {
  const user_id = userIdInput.value.trim();
  const message = messageInput.value.trim();

  if (!user_id || !message) {
    alert("Please enter a user id and a message.");
    return;
  }

  addMessage(message, "user", "You");
  messageInput.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id,
      session_id: activeSessionId,
      message
    })
  });

  const data = await res.json();

  activeSessionId = data.session_id;
  addMessage(data.response || "", "bot", "Bot");

  if (data.summary) {
    summaryBox.textContent = "Summary: " + data.summary;
    summaryBox.style.display = "block";
  }

  await loadSessions();
}

async function queryHistory() {
  const user_id = userIdInput.value.trim();
  const query = queryInput.value.trim();

  if (!user_id || !query) {
    alert("Please enter a user id and a query.");
    return;
  }

  const res = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id, query })
  });

  const data = await res.json();

  ragResults.innerHTML = `<strong>Answer:</strong> ${data.answer}`;

  if (data.context && data.context.length) {
    const contextHtml = data.context
      .map(ctx => `<strong>${ctx.day}</strong> (${ctx.similarity}): ${ctx.summary}`)
      .join("<br/>");
    ragResults.innerHTML += `<div class="rag-context"><strong>Sources:</strong><br/>${contextHtml}</div>`;
  }

  ragResults.classList.add("show");
  ragResults.style.display = "block";
}

sendBtn.addEventListener("click", sendMessage);
queryBtn.addEventListener("click", queryHistory);
clearBtn.addEventListener("click", clearChat);

prevMonthBtn.addEventListener("click", () => {
  currentDate.setMonth(currentDate.getMonth() - 1);
  renderCalendar();
});

nextMonthBtn.addEventListener("click", () => {
  currentDate.setMonth(currentDate.getMonth() + 1);
  renderCalendar();
});

messageInput.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

loadSessions();