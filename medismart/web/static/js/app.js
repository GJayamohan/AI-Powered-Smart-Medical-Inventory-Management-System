/**
 * MediSmart Client-Side Interactive JavaScript Application.
 */

// Utility: Format Currency (INR / Rs)
function formatCurrency(val) {
  if (val === null || val === undefined) return "Rs 0.00";
  return "Rs " + Number(val).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// Utility: Format Date
function formatDate(dateStr) {
  if (!dateStr) return "-";
  const d = new Date(dateStr);
  return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
}

// Utility: Toast Notification
function showToast(message, type = "info") {
  const container = document.getElementById("toast-container") || createToastContainer();
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.style.cssText = `
    background: ${type === 'danger' ? '#ef4444' : type === 'success' ? '#10b981' : '#0f766e'};
    color: #fff;
    padding: 12px 18px;
    border-radius: 8px;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    margin-top: 10px;
    font-size: 13px;
    font-weight: 500;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    gap: 8px;
  `;
  toast.innerHTML = `<span>${type === 'success' ? '✓' : type === 'danger' ? '⚠' : 'ℹ'}</span> ${message}`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

function createToastContainer() {
  const c = document.createElement("div");
  c.id = "toast-container";
  c.style.cssText = "position: fixed; top: 20px; right: 20px; z-index: 9999;";
  document.body.appendChild(c);
  return c;
}

// Global Poll: Update Topbar Badges
async function updateTopbarStats() {
  try {
    const res = await fetch("/api/dashboard/summary");
    if (!res.ok) return;
    const data = await res.json();
    if (!data.success) return;

    const nearExpBadge = document.getElementById("topbar-near-expiry-badge");
    const lowStockBadge = document.getElementById("topbar-low-stock-badge");

    if (nearExpBadge) {
      const count = data.expiry_risk.near_expiry_count;
      nearExpBadge.textContent = `${count} Near Expiry`;
      if (count > 0) nearExpBadge.classList.add("alert-active");
      else nearExpBadge.classList.remove("alert-active");
    }

    if (lowStockBadge) {
      const count = data.inventory.low_stock_count;
      lowStockBadge.textContent = `${count} Low Stock`;
      if (count > 0) lowStockBadge.classList.add("alert-active");
      else lowStockBadge.classList.remove("alert-active");
    }
  } catch (err) {
    console.debug("Error polling stats:", err);
  }
}

// Modal Helpers
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.add("active");
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.remove("active");
}

// Close modals on clicking overlay
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".modal-overlay").forEach((overlay) => {
    overlay.addEventListener("click", (e) => {
      if (e.target === overlay) {
        overlay.classList.remove("active");
      }
    });
  });

  // Initial poll for alerts
  updateTopbarStats();
  setInterval(updateTopbarStats, 30000); // 30s refresh
});

// Chatbot Interactive Assistant
function toggleChatbot() {
  const win = document.getElementById("chatbot-window");
  if (win) {
    win.classList.toggle("open");
    if (win.classList.contains("open")) {
      document.getElementById("chat-user-input")?.focus();
    }
  }
}

async function sendChatMessage() {
  const input = document.getElementById("chat-user-input");
  const msg = input.value.trim();
  if (!msg) return;

  const body = document.getElementById("chatbot-body");

  // Append user message
  const userDiv = document.createElement("div");
  userDiv.className = "chat-msg user";
  userDiv.textContent = msg;
  body.appendChild(userDiv);
  input.value = "";
  body.scrollTop = body.scrollHeight;

  // Typing indicator
  const typingDiv = document.createElement("div");
  typingDiv.className = "chat-msg bot";
  typingDiv.textContent = "Analyzing inventory & clinical knowledge...";
  body.appendChild(typingDiv);
  body.scrollTop = body.scrollHeight;

  try {
    const res = await fetch("/api/chat/message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: msg }),
    });
    const data = await res.json();
    typingDiv.remove();

    const botDiv = document.createElement("div");
    botDiv.className = "chat-msg bot";
    // Convert markdown bold to html bold
    let replyHtml = data.reply
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");
    botDiv.innerHTML = replyHtml;
    body.appendChild(botDiv);
  } catch (err) {
    typingDiv.remove();
    const botDiv = document.createElement("div");
    botDiv.className = "chat-msg bot";
    botDiv.textContent = "Sorry, I encountered an error connecting to the knowledge service.";
    body.appendChild(botDiv);
  }
  body.scrollTop = body.scrollHeight;
}
