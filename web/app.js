// Control-LLM Web Console Client Logic (Streaming, KaTeX Math Protection, Font Size, Python Plots)

document.addEventListener('DOMContentLoaded', () => {
  const sidebar = document.getElementById('sidebar');
  const btnSidebarToggle = document.getElementById('btn-sidebar-toggle');
  const btnNewChat = document.getElementById('btn-new-chat');
  const historyList = document.getElementById('history-list');
  const chatViewport = document.getElementById('chat-viewport') || document.getElementById('chat-thread');
  const chatThread = document.getElementById('chat-thread');
  const chatForm = document.getElementById('chat-form');
  const chatInput = document.getElementById('chat-input');
  const btnSend = document.getElementById('btn-send');
  const btnThemeToggle = document.getElementById('btn-theme-toggle');

  // File Attachment Elements
  const btnAttachFile = document.getElementById('btn-attach-file');
  const chatFileInput = document.getElementById('chat-file-input');
  const attachmentPreview = document.getElementById('attachment-preview');
  const attachmentName = document.getElementById('attachment-name');
  const btnRemoveAttachment = document.getElementById('btn-remove-attachment');

  let attachedFile = null;
  let isGenerating = false;

  // Modals
  const btnUploadModal = document.getElementById('btn-upload-modal');
  const uploadModal = document.getElementById('upload-modal');
  const btnCloseUploadModal = document.getElementById('btn-close-upload-modal');
  const modalDropZone = document.getElementById('modal-drop-zone');
  const modalFileInput = document.getElementById('modal-file-input');
  const modalUploadStatus = document.getElementById('modal-upload-status');

  const btnSettingsModal = document.getElementById('btn-settings-modal');
  const settingsModal = document.getElementById('settings-modal');
  const btnCloseSettingsModal = document.getElementById('btn-close-settings-modal');
  const btnClearChats = document.getElementById('btn-clear-chats');

  // Font Size Management
  const savedFontSize = localStorage.getItem('control_llm_fontsize') || '15px';
  document.documentElement.style.setProperty('--chat-font-size', savedFontSize);

  document.querySelectorAll('.btn-font-size').forEach((btn) => {
    const size = btn.getAttribute('data-size');
    if (size === savedFontSize) btn.classList.add('active');

    btn.addEventListener('click', () => {
      document.querySelectorAll('.btn-font-size').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      document.documentElement.style.setProperty('--chat-font-size', size);
      localStorage.setItem('control_llm_fontsize', size);
    });
  });

  // Session Data in LocalStorage
  const STORAGE_KEY = 'control_llm_sessions_v1';
  let sessions = loadSessions();
  let currentSessionId = null;

  function loadSessions() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || [];
    } catch {
      return [];
    }
  }

  function saveSessions() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
  }

  // Theme Management
  const savedTheme = localStorage.getItem('control_llm_theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);

  btnThemeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', nextTheme);
    localStorage.setItem('control_llm_theme', nextTheme);
  });

  // Sidebar Toggle
  btnSidebarToggle.addEventListener('click', () => {
    sidebar.classList.toggle('collapsed');
  });

  // Configure Marked with Highlight.js and Custom Code Blocks
  const customRenderer = new marked.Renderer();
  customRenderer.code = function(code, lang) {
    let highlighted = code;
    const validLang = lang && window.hljs && hljs.getLanguage(lang) ? lang : '';
    if (window.hljs) {
      try {
        highlighted = validLang 
          ? hljs.highlight(code, { language: validLang, ignoreIllegals: true }).value 
          : hljs.highlightAuto(code).value;
      } catch (e) {
        highlighted = code;
      }
    }
    const displayLang = (validLang || lang || 'code').toUpperCase();
    const encodedCode = encodeURIComponent(code);
    return `
      <div class="code-block-wrapper">
        <div class="code-block-header">
          <span class="code-lang-label">${displayLang}</span>
          <button type="button" class="btn-copy-code" data-code="${encodedCode}" title="Copy code">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
            <span>Copy</span>
          </button>
        </div>
        <pre><code class="hljs ${validLang}">${highlighted}</code></pre>
      </div>
    `;
  };

  marked.setOptions({
    breaks: true,
    gfm: true,
    renderer: customRenderer,
  });

  // Global Copy Code Handler
  document.addEventListener('click', (e) => {
    const copyBtn = e.target.closest('.btn-copy-code');
    if (copyBtn) {
      const codeText = decodeURIComponent(copyBtn.getAttribute('data-code') || '');
      navigator.clipboard.writeText(codeText).then(() => {
        const origHtml = copyBtn.innerHTML;
        copyBtn.innerHTML = `
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>
          <span>Copied!</span>
        `;
        setTimeout(() => {
          copyBtn.innerHTML = origHtml;
        }, 2000);
      });
    }
  });

  // Robust Markdown + KaTeX Renderer (Protects all Math delimiters before Markdown processing)
  function renderMarkdownWithKaTeX(rawText) {
    if (!rawText) return '';

    const mathPlaceholders = [];
    let text = rawText;

    // Clean up any raw LaTeX document structure artifacts
    text = text.replace(/\\section\*?\{([\s\S]*?)\}/g, '### $1\n\n');
    text = text.replace(/\\subsection\*?\{([\s\S]*?)\}/g, '#### $1\n\n');
    text = text.replace(/\\subsubsection\*?\{([\s\S]*?)\}/g, '##### $1\n\n');
    text = text.replace(/\\begin\{figure\}[\s\S]*?\\end\{figure\}/g, (match) => {
      const capMatch = match.match(/\\caption\{([\s\S]*?)\}/);
      const imgMatch = match.match(/\\includegraphics.*?\{([^\}]+)\}/);
      let out = '';
      if (imgMatch) {
        const fname = imgMatch[1].trim();
        out += `\n\n![${capMatch ? capMatch[1] : 'Simulation Plot'}](/plots/${fname})\n\n`;
      }
      if (capMatch && !imgMatch) {
        out += `\n\n*${capMatch[1]}*\n\n`;
      }
      return out;
    });
    text = text.replace(/\\(centering|begin\{center\}|end\{center\})/g, '');
    text = text.replace(/\\includegraphics.*?\{([^\}]+)\}/g, '![$1](/plots/$1)');
    text = text.replace(/\\caption\{([\s\S]*?)\}/g, '*$1*');

    // 0. Extract LaTeX Environments: \begin{aligned}...\end{aligned}, \begin{bmatrix}...\end{bmatrix}, etc.
    text = text.replace(/\\begin\{(aligned|bmatrix|matrix|pmatrix|vmatrix|cases|equation\*?)\}[\s\S]*?\\end\{\1\}/g, (match) => {
      const ph = `KATEXBLOCK${mathPlaceholders.length}KATEX`;
      mathPlaceholders.push({ type: 'block', formula: match.trim() });
      return `\n\n${ph}\n\n`;
    });

    // 1. Extract Display Math: $$ ... $$
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, (match, formula) => {
      const ph = `KATEXBLOCK${mathPlaceholders.length}KATEX`;
      mathPlaceholders.push({ type: 'block', formula: formula.trim() });
      return `\n\n${ph}\n\n`;
    });

    // 2. Extract Display Math: \[ ... \]
    text = text.replace(/\\\[([\s\S]*?)\\\]/g, (match, formula) => {
      const ph = `KATEXBLOCK${mathPlaceholders.length}KATEX`;
      mathPlaceholders.push({ type: 'block', formula: formula.trim() });
      return `\n\n${ph}\n\n`;
    });

    // 3. Extract Inline Math: $ ... $ (excluding empty or multi-line)
    text = text.replace(/\$([^\$\n]+?)\$/g, (match, formula) => {
      const ph = `KATEXINLINE${mathPlaceholders.length}KATEX`;
      mathPlaceholders.push({ type: 'inline', formula: formula.trim() });
      return ph;
    });

    // 4. Extract Inline Math: \( ... \)
    text = text.replace(/\\\(([\s\S]*?)\\\)/g, (match, formula) => {
      const ph = `KATEXINLINE${mathPlaceholders.length}KATEX`;
      mathPlaceholders.push({ type: 'inline', formula: formula.trim() });
      return ph;
    });

    // Parse Markdown
    let html = marked.parse(text);

    // Replace placeholders with KaTeX rendered HTML
    mathPlaceholders.forEach((item, index) => {
      const ph = item.type === 'block' ? `KATEXBLOCK${index}KATEX` : `KATEXINLINE${index}KATEX`;
      let renderedMath = item.formula;
      if (window.katex) {
        try {
          renderedMath = window.katex.renderToString(item.formula, {
            displayMode: item.type === 'block',
            throwOnError: false,
          });
        } catch (e) {
          renderedMath = item.formula;
        }
      }
      const regex = new RegExp(`(<p>\\s*)?${ph}(\\s*<\\/p>)?`, 'g');
      html = html.replace(regex, renderedMath);
      html = html.split(ph).join(renderedMath);
    });

    return html;
  }

  // Math Helper Toolbar
  document.querySelectorAll('.math-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const snippet = btn.getAttribute('data-insert');
      if (snippet) insertSnippetAtCursor(snippet);
    });
  });

  function insertSnippetAtCursor(snippet) {
    const startPos = chatInput.selectionStart || 0;
    const endPos = chatInput.selectionEnd || 0;
    const currentText = chatInput.value;

    chatInput.value = currentText.substring(0, startPos) + snippet + currentText.substring(endPos);
    chatInput.focus();
    chatInput.selectionStart = startPos + snippet.length;
    chatInput.selectionEnd = startPos + snippet.length;

    chatInput.style.height = 'auto';
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
  }

  // File Attachment Handling
  btnAttachFile.addEventListener('click', () => chatFileInput.click());

  chatFileInput.addEventListener('change', () => {
    if (chatFileInput.files.length > 0) {
      const file = chatFileInput.files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        attachedFile = {
          name: file.name,
          content: e.target.result,
        };
        attachmentName.textContent = file.name;
        attachmentPreview.style.display = 'flex';
      };
      reader.readAsText(file);
    }
  });

  btnRemoveAttachment.addEventListener('click', () => {
    attachedFile = null;
    attachmentPreview.style.display = 'none';
    chatFileInput.value = '';
  });

  // History List
  function renderHistorySidebar() {
    historyList.innerHTML = '';
    if (sessions.length === 0) {
      historyList.innerHTML = '<div style="font-size: 12px; color: var(--text-muted); padding: 6px 10px;">No chats yet</div>';
      return;
    }

    sessions.forEach((s) => {
      const item = document.createElement('div');
      item.className = `history-item ${s.id === currentSessionId ? 'active' : ''}`;

      const titleSpan = document.createElement('span');
      titleSpan.className = 'history-item-title';
      titleSpan.textContent = s.title || 'Untitled Session';
      titleSpan.title = s.title;

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'btn-delete-chat';
      deleteBtn.innerHTML = '&times;';
      deleteBtn.title = 'Delete chat';
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteSession(s.id);
      });

      item.appendChild(titleSpan);
      item.appendChild(deleteBtn);
      item.addEventListener('click', () => switchSession(s.id));
      historyList.appendChild(item);
    });
  }

  function startNewSession() {
    currentSessionId = `session_${Date.now()}`;
    const newSession = {
      id: currentSessionId,
      title: 'New Chat',
      createdAt: Date.now(),
      messages: [],
    };
    sessions.unshift(newSession);
    saveSessions();
    renderHistorySidebar();
    renderActiveSession();
  }

  function switchSession(id) {
    currentSessionId = id;
    renderHistorySidebar();
    renderActiveSession();
  }

  function deleteSession(id) {
    sessions = sessions.filter((s) => s.id !== id);
    saveSessions();
    if (currentSessionId === id) {
      currentSessionId = sessions.length > 0 ? sessions[0].id : null;
      if (!currentSessionId) {
        startNewSession();
        return;
      }
    }
    renderHistorySidebar();
    renderActiveSession();
  }

  function renderActiveSession() {
    chatThread.innerHTML = '';
    const current = sessions.find((s) => s.id === currentSessionId);

    if (!current || current.messages.length === 0) {
      chatThread.innerHTML = `
        <div class="hero-container" id="hero-container">
          <div class="hero-title">ControlAI</div>
          <div class="hero-subtitle">Control Systems &bull; Computational Python &bull; Mathematical Simulation</div>

          <div class="preset-cards">
            <div class="preset-card" data-prompt="Design an LQR controller for A=[[0, 1], [-2, -3]], B=[[0], [1]], Q=diag([10, 1]), R=1 and simulate the step response.">
              <div class="preset-heading">Optimal LQR Synthesis</div>
              <div class="preset-detail">Solve CARE, calculate feedback gain K, and simulate closed-loop response.</div>
            </div>
            <div class="preset-card" data-prompt="Write and run a Python script to simulate and plot the step response of a second order system with damping ratio zeta=0.6 and natural frequency wn=1.4 rad/s.">
              <div class="preset-heading">Second-Order Simulation</div>
              <div class="preset-detail">Execute Python ODE simulation and generate step response plot.</div>
            </div>
            <div class="preset-card" data-prompt="Compute the Gain Margin, Phase Margin, and crossover frequencies for G(s) = 10 / (s*(s+1)*(s+5)).">
              <div class="preset-heading">Stability Margins</div>
              <div class="preset-detail">Frequency response, phase margin, gain margin, and Bode plot.</div>
            </div>
            <div class="preset-card" data-prompt="Discretize A=[[0, 1], [-4, -2]], B=[[0], [2]] with Ts=0.05s using exact Zero-Order Hold (ZOH).">
              <div class="preset-heading">Exact ZOH Discretization</div>
              <div class="preset-detail">Matrix exponential Van Loan calculation for discrete Ad, Bd.</div>
            </div>
          </div>
        </div>
      `;
      attachPresetEvents();
      return;
    }

    current.messages.forEach((msg) => {
      renderMessage(msg.role, msg.content, msg.tool_traces, msg.plots, msg.thoughts);
    });

    chatViewport.scrollTop = chatViewport.scrollHeight;
  }

  function renderThoughtBox(thoughts, isOpen = false) {
    if (!thoughts || thoughts.length === 0) return '';
    const thoughtList = Array.isArray(thoughts) ? thoughts : [thoughts];
    const text = thoughtList.join('\n\n');
    return `
      <details class="thought-accordion" ${isOpen ? 'open' : ''}>
        <summary class="thought-summary">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
          <span>Thinking Process (${thoughtList.length} steps)</span>
        </summary>
        <div class="thought-content">${escapeHtml(text)}</div>
      </details>
    `;
  }

  function renderMessage(role, rawContent, toolTraces = [], plots = [], thoughts = []) {
    const isUser = role === 'user';
    const label = isUser ? 'You' : 'ControlAI';
    const entry = document.createElement('div');
    entry.className = `message-entry ${isUser ? 'user' : 'bot'}`;

    const lbl = document.createElement('div');
    lbl.className = 'message-label';
    lbl.textContent = label;

    const content = document.createElement('div');
    content.className = 'message-content';

    let html = '';
    if (isUser) {
      html = escapeHtml(rawContent);
    } else {
      if (thoughts && thoughts.length > 0) {
        html += renderThoughtBox(thoughts, false);
      }
      let cleanText = (rawContent || '').replace(/\{\s*["']name["']\s*:[\s\S]*?\}\s*\}/g, '').trim();
      cleanText = cleanText.replace(/<tool_call>[\s\S]*?<\/tool_call>/g, '').trim();
      if (cleanText) {
        html += `<div class="response-body">${renderMarkdownWithKaTeX(cleanText)}</div>`;
      }
    }

    content.innerHTML = html;

    if (!isUser) {
      if (plots && plots.length > 0) {
        plots.forEach((url) => {
          const plotDiv = document.createElement('div');
          plotDiv.className = 'plot-container';

          const img = document.createElement('img');
          img.className = 'plot-image';
          img.src = url;
          img.alt = 'Simulation Plot';

          const cap = document.createElement('div');
          cap.className = 'plot-caption';
          cap.textContent = `Plot: ${url.split('/').pop()}`;

          plotDiv.appendChild(img);
          plotDiv.appendChild(cap);
          content.appendChild(plotDiv);
        });
      }
    }

    entry.appendChild(lbl);
    entry.appendChild(content);
    chatThread.appendChild(entry);
    return entry;
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  function attachPresetEvents() {
    document.querySelectorAll('.preset-card').forEach((card) => {
      card.addEventListener('click', () => {
        const prompt = card.getAttribute('data-prompt');
        if (prompt) {
          chatInput.value = prompt;
          sendMessage(prompt);
        }
      });
    });
  }

  // Modals Handling
  btnUploadModal.addEventListener('click', () => {
    uploadModal.classList.add('active');
    modalUploadStatus.textContent = '';
  });
  btnCloseUploadModal.addEventListener('click', () => uploadModal.classList.remove('active'));

  btnSettingsModal.addEventListener('click', () => settingsModal.classList.add('active'));
  btnCloseSettingsModal.addEventListener('click', () => settingsModal.classList.remove('active'));

  window.addEventListener('click', (e) => {
    if (e.target === uploadModal) uploadModal.classList.remove('active');
    if (e.target === settingsModal) settingsModal.classList.remove('active');
  });

  btnClearChats.addEventListener('click', () => {
    if (confirm('Clear all chat history?')) {
      sessions = [];
      saveSessions();
      startNewSession();
      settingsModal.classList.remove('active');
    }
  });

  // Modal File Upload
  modalDropZone.addEventListener('click', () => modalFileInput.click());
  modalDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    modalDropZone.classList.add('dragover');
  });
  modalDropZone.addEventListener('dragleave', () => modalDropZone.classList.remove('dragover'));
  modalDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    modalDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) handleModalFileUpload(e.dataTransfer.files[0]);
  });
  modalFileInput.addEventListener('change', () => {
    if (modalFileInput.files.length > 0) handleModalFileUpload(modalFileInput.files[0]);
  });

  async function handleModalFileUpload(file) {
    modalUploadStatus.style.color = 'var(--text-accent)';
    modalUploadStatus.textContent = `Indexing ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        modalUploadStatus.style.color = 'var(--text-accent)';
        modalUploadStatus.textContent = `Indexed: +${data.chunks_added} chunks (${data.pages_parsed} pages)`;
        setTimeout(() => uploadModal.classList.remove('active'), 1800);
      } else {
        modalUploadStatus.style.color = 'var(--text-danger)';
        modalUploadStatus.textContent = `Error: ${data.detail || 'Upload failed'}`;
      }
    } catch (err) {
      modalUploadStatus.style.color = 'var(--text-danger)';
      modalUploadStatus.textContent = `Error: ${err.message}`;
    }
  }

  // New Chat Click
  btnNewChat.addEventListener('click', () => startNewSession());

  // Textarea Auto-resize
  chatInput.addEventListener('input', () => {
    chatInput.style.height = 'auto';
    chatInput.style.height = `${Math.min(chatInput.scrollHeight, 160)}px`;
  });

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (chatInput.value.trim() && !isGenerating) {
        chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  // Submit Handler
  chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const text = chatInput.value.trim();
    if (!text || isGenerating) return;
    sendMessage(text);
  });

  // Smart Scroll Detection (allows user to scroll up freely without fighting the stream)
  let userScrolledUp = false;

  chatViewport.addEventListener('scroll', () => {
    const distanceFromBottom = chatViewport.scrollHeight - chatViewport.scrollTop - chatViewport.clientHeight;
    userScrolledUp = distanceFromBottom > 90;
  });

  function smartScroll() {
    if (!userScrolledUp) {
      chatViewport.scrollTop = chatViewport.scrollHeight;
    }
  }

  // Real-Time Token Streaming Message Sender
  async function sendMessage(promptText) {
    const hero = document.getElementById('hero-container');
    if (hero) hero.style.display = 'none';

    let current = sessions.find((s) => s.id === currentSessionId);
    if (!current) {
      startNewSession();
      current = sessions.find((s) => s.id === currentSessionId);
    }

    if (current.messages.length === 0) {
      current.title = promptText.length > 28 ? promptText.substring(0, 28) + '...' : promptText;
      renderHistorySidebar();
    }

    let fullPrompt = promptText;
    if (attachedFile) {
      fullPrompt += `\n\n[Attached File: ${attachedFile.name}]\n\`\`\`\n${attachedFile.content}\n\`\`\``;
      attachedFile = null;
      attachmentPreview.style.display = 'none';
      chatFileInput.value = '';
    }

    // Reset scroll lock on new user message
    userScrolledUp = false;

    // Append User Message
    current.messages.push({ role: 'user', content: fullPrompt, tool_traces: [], plots: [] });
    saveSessions();
    renderMessage('user', fullPrompt);

    chatInput.value = '';
    chatInput.style.height = 'auto';

    // Create Bot Message Box for live streaming
    const botEntry = document.createElement('div');
    botEntry.className = 'message-entry bot';
    botEntry.innerHTML = `
      <div class="message-label">ControlAI</div>
      <div class="message-content"><span class="generating-indicator">Working...</span></div>
    `;
    chatThread.appendChild(botEntry);
    smartScroll();

    const contentBox = botEntry.querySelector('.message-content');

    isGenerating = true;
    btnSend.disabled = true;

    let accumulatedText = '';
    let toolTraces = [];
    let plots = [];
    let thoughts = [];

    // Build conversation history (excluding the user message we just added)
    const historyPayload = current.messages.slice(0, -1).map((m) => ({
      role: m.role === 'user' ? 'user' : 'assistant',
      content: m.content,
    }));

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: fullPrompt, history: historyPayload }),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop(); // Keep partial line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.replace('data: ', '').trim();
          if (!jsonStr) continue;

          try {
            const event = JSON.parse(jsonStr);

            if (event.type === 'thought') {
              thoughts.push(event.content);
              const thoughtHtml = renderThoughtBox(thoughts, true);
              const textHtml = accumulatedText 
                ? `<div class="response-body">${renderMarkdownWithKaTeX(accumulatedText)}<span class="streaming-cursor"></span></div>` 
                : `<div class="response-body"><span class="generating-indicator">Working...</span><span class="streaming-cursor"></span></div>`;
              contentBox.innerHTML = thoughtHtml + textHtml;
              smartScroll();
            } else if (event.type === 'token') {
              accumulatedText += event.content;
              const thoughtHtml = renderThoughtBox(thoughts, false);
              contentBox.innerHTML = thoughtHtml + `<div class="response-body">${renderMarkdownWithKaTeX(accumulatedText)}<span class="streaming-cursor"></span></div>`;
              smartScroll();
            } else if (event.type === 'tool_start') {
              // Tool starting indicator
            } else if (event.type === 'tool_end') {
              toolTraces.push(event.trace);
            } else if (event.type === 'plot') {
              if (!plots.includes(event.url)) plots.push(event.url);
            } else if (event.type === 'done') {
              accumulatedText = event.response || accumulatedText;
              if (event.traces) toolTraces = event.traces;
              if (event.plots) plots = event.plots;
              if (event.thoughts) thoughts = event.thoughts;
            } else if (event.type === 'error') {
              throw new Error(event.error);
            }
          } catch (e) {
            console.error('Parse SSE event error:', e);
          }
        }
      }

      // Final Render without cursor
      const finalThoughtHtml = renderThoughtBox(thoughts, false);
      contentBox.innerHTML = finalThoughtHtml + `<div class="response-body">${renderMarkdownWithKaTeX(accumulatedText)}</div>`;

      // Append plots
      if (plots.length > 0) {
        plots.forEach((url) => {
          const plotDiv = document.createElement('div');
          plotDiv.className = 'plot-container';

          const img = document.createElement('img');
          img.className = 'plot-image';
          img.src = url;
          img.alt = 'Simulation Plot';

          const cap = document.createElement('div');
          cap.className = 'plot-caption';
          cap.textContent = `Plot: ${url.split('/').pop()}`;

          plotDiv.appendChild(img);
          plotDiv.appendChild(cap);
          contentBox.appendChild(plotDiv);
        });
      }

      // Save assistant message to session
      current.messages.push({
        role: 'assistant',
        content: accumulatedText,
        tool_traces: toolTraces,
        plots: plots,
        thoughts: thoughts,
      });
      saveSessions();

    } catch (err) {
      contentBox.innerHTML = `<span style="color: var(--text-danger);">Execution Error: ${err.message}</span>`;
    } finally {
      isGenerating = false;
      btnSend.disabled = false;
      smartScroll();
    }
  }

  // Initialize
  if (sessions.length === 0) {
    startNewSession();
  } else {
    currentSessionId = sessions[0].id;
    renderHistorySidebar();
    renderActiveSession();
  }
});
