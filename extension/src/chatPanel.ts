import * as vscode from 'vscode';

export class ChatPanel {
    private panel: vscode.WebviewPanel | undefined;
    private context: vscode.ExtensionContext;
    private conversationId: string | undefined;
    private messages: Array<{ role: string; content: string }> = [];

    constructor(context: vscode.ExtensionContext) {
        this.context = context;
    }

    reveal() {
        if (this.panel) {
            this.panel.reveal();
            return;
        }

        this.panel = vscode.window.createWebviewPanel(
            'omniscientChat',
            'OmniscientLLM Chat',
            vscode.ViewColumn.Beside,
            {
                enableScripts: true,
                retainContextWhenHidden: true,
            }
        );

        this.panel.webview.html = this.getWebviewContent();

        this.panel.webview.onDidReceiveMessage(
            async (message) => {
                switch (message.command) {
                    case 'sendMessage':
                        await this.handleUserMessage(message.text);
                        break;
                    case 'newConversation':
                        this.messages = [];
                        this.conversationId = undefined;
                        break;
                    case 'copyCode':
                        await vscode.env.clipboard.writeText(message.code);
                        vscode.window.showInformationMessage('Code copied to clipboard');
                        break;
                }
            },
            undefined,
            this.context.subscriptions
        );

        this.panel.onDidDispose(() => {
            this.panel = undefined;
        });
    }

    async sendMessage(text: string) {
        this.reveal();
        // Wait a tick for panel to initialize
        await new Promise(resolve => setTimeout(resolve, 100));
        await this.handleUserMessage(text);
    }

    private async handleUserMessage(text: string) {
        const config = vscode.workspace.getConfiguration('omniscient');
        const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');
        const enableThinking = config.get<boolean>('enableThinking', false);
        const maxTokens = config.get<number>('maxTokens', 512);

        // Auto-attach selected code
        const context = this.getAutoContext();
        const fullMessage = context ? `${text}\n${context}` : text;

        this.messages.push({ role: 'user', content: fullMessage });

        // Show user message in webview
        this.panel?.webview.postMessage({
            type: 'userMessage',
            content: text,
            context: context || undefined,
        });

        // Stream response from server
        try {
            const response = await fetch(`${serverUrl}/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    messages: this.messages,
                    max_tokens: maxTokens,
                    temperature: 0.3,
                    stream: true,
                    enable_thinking: enableThinking,
                    conversation_id: this.conversationId,
                }),
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const reader = response.body?.getReader();
            if (!reader) throw new Error('No response body');

            const decoder = new TextDecoder();
            let fullContent = '';

            // Signal start of assistant message
            this.panel?.webview.postMessage({ type: 'assistantStart' });

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    const data = line.slice(6).trim();
                    if (data === '[DONE]') continue;

                    try {
                        const parsed = JSON.parse(data);
                        const content = parsed.choices?.[0]?.delta?.content;
                        if (content) {
                            fullContent += content;
                            this.panel?.webview.postMessage({
                                type: 'assistantChunk',
                                content,
                            });
                        }
                    } catch {
                        // Skip malformed chunks
                    }
                }
            }

            this.panel?.webview.postMessage({ type: 'assistantEnd' });
            this.messages.push({ role: 'assistant', content: fullContent });

        } catch (error: any) {
            this.panel?.webview.postMessage({
                type: 'error',
                content: `Error: ${error.message}`,
            });
        }
    }

    private getAutoContext(): string {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return '';

        const selection = editor.selection;
        if (selection.isEmpty) return '';

        const doc = editor.document;
        const selectedText = doc.getText(selection);
        const fileName = doc.fileName.split('/').pop() || doc.fileName;
        const startLine = selection.start.line + 1;

        return `\n[Selected code from ${fileName}:${startLine}]\n\`\`\`${doc.languageId}\n${selectedText}\n\`\`\``;
    }

    dispose() {
        this.panel?.dispose();
    }

    private getWebviewContent(): string {
        return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body {
        font-family: var(--vscode-font-family, 'Segoe UI', sans-serif);
        font-size: var(--vscode-font-size, 13px);
        color: var(--vscode-foreground, #ccc);
        background: var(--vscode-editor-background, #1e1e1e);
        display: flex;
        flex-direction: column;
        height: 100vh;
    }
    #toolbar {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-bottom: 1px solid var(--vscode-panel-border, #333);
    }
    #toolbar button {
        background: var(--vscode-button-background, #0e639c);
        color: var(--vscode-button-foreground, #fff);
        border: none;
        padding: 4px 10px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 12px;
    }
    #toolbar label {
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    #messages {
        flex: 1;
        overflow-y: auto;
        padding: 12px;
    }
    .message {
        margin-bottom: 16px;
        padding: 8px 12px;
        border-radius: 6px;
        max-width: 90%;
        line-height: 1.5;
    }
    .message.user {
        background: var(--vscode-input-background, #2d2d2d);
        margin-left: auto;
    }
    .message.assistant {
        background: var(--vscode-editor-inactiveSelectionBackground, #264f78);
    }
    .message.error {
        background: #5a1d1d;
        color: #f88;
    }
    .message .role {
        font-weight: bold;
        font-size: 11px;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 4px;
    }
    .message pre {
        background: #0d0d0d;
        padding: 8px;
        border-radius: 4px;
        overflow-x: auto;
        margin: 6px 0;
        position: relative;
    }
    .message pre code {
        font-family: var(--vscode-editor-font-family, 'Consolas', monospace);
        font-size: 12px;
    }
    .copy-btn {
        position: absolute;
        top: 4px;
        right: 4px;
        background: #444;
        color: #ccc;
        border: none;
        padding: 2px 6px;
        border-radius: 3px;
        cursor: pointer;
        font-size: 11px;
    }
    .copy-btn:hover { background: #666; }
    .context-tag {
        font-size: 11px;
        opacity: 0.6;
        font-style: italic;
        margin-bottom: 4px;
    }
    #input-area {
        display: flex;
        gap: 8px;
        padding: 10px 12px;
        border-top: 1px solid var(--vscode-panel-border, #333);
    }
    #input-area textarea {
        flex: 1;
        background: var(--vscode-input-background, #2d2d2d);
        color: var(--vscode-input-foreground, #ccc);
        border: 1px solid var(--vscode-input-border, #444);
        border-radius: 4px;
        padding: 8px;
        font-family: inherit;
        font-size: 13px;
        resize: none;
        min-height: 40px;
        max-height: 120px;
    }
    #input-area button {
        background: var(--vscode-button-background, #0e639c);
        color: var(--vscode-button-foreground, #fff);
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        align-self: flex-end;
    }
</style>
</head>
<body>
    <div id="toolbar">
        <button id="newChat">New Chat</button>
        <label>
            <input type="checkbox" id="thinkToggle"> Show Thinking
        </label>
    </div>
    <div id="messages"></div>
    <div id="input-area">
        <textarea id="userInput" placeholder="Ask about code..." rows="2"></textarea>
        <button id="sendBtn">Send</button>
    </div>

<script>
    const vscode = acquireVsCodeApi();
    const messagesEl = document.getElementById('messages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChat');
    let currentAssistantEl = null;
    let currentContent = '';

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function renderContent(text) {
        // Simple markdown: code blocks and inline code
        let html = escapeHtml(text);
        // Code blocks
        html = html.replace(/\\\`\\\`\\\`(\\w*?)\\n([\\s\\S]*?)\\\`\\\`\\\`/g, (_, lang, code) => {
            const id = 'code-' + Math.random().toString(36).substr(2, 9);
            return '<pre><code class="language-' + lang + '" id="' + id + '">' + code + '</code><button class="copy-btn" onclick="copyCode(\\'' + id + '\\')">Copy</button></pre>';
        });
        // Inline code
        html = html.replace(/\\\`([^\\\`]+)\\\`/g, '<code>$1</code>');
        // Bold
        html = html.replace(/\\*\\*(.+?)\\*\\*/g, '<strong>$1</strong>');
        // Newlines
        html = html.replace(/\\n/g, '<br>');
        return html;
    }

    function addMessage(role, content, context) {
        const div = document.createElement('div');
        div.className = 'message ' + role;
        let html = '<div class="role">' + role + '</div>';
        if (context) {
            html += '<div class="context-tag">Code attached</div>';
        }
        html += '<div class="content">' + renderContent(content) + '</div>';
        div.innerHTML = html;
        messagesEl.appendChild(div);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
    }

    function send() {
        const text = userInput.value.trim();
        if (!text) return;
        userInput.value = '';
        vscode.postMessage({ command: 'sendMessage', text });
    }

    sendBtn.addEventListener('click', send);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            send();
        }
    });

    newChatBtn.addEventListener('click', () => {
        messagesEl.innerHTML = '';
        vscode.postMessage({ command: 'newConversation' });
    });

    window.copyCode = function(id) {
        const el = document.getElementById(id);
        if (el) {
            vscode.postMessage({ command: 'copyCode', code: el.textContent });
        }
    };

    window.addEventListener('message', (event) => {
        const msg = event.data;
        switch (msg.type) {
            case 'userMessage':
                addMessage('user', msg.content, msg.context);
                break;
            case 'assistantStart':
                currentContent = '';
                currentAssistantEl = addMessage('assistant', '...');
                break;
            case 'assistantChunk':
                currentContent += msg.content;
                if (currentAssistantEl) {
                    const contentEl = currentAssistantEl.querySelector('.content');
                    if (contentEl) contentEl.innerHTML = renderContent(currentContent);
                    messagesEl.scrollTop = messagesEl.scrollHeight;
                }
                break;
            case 'assistantEnd':
                currentAssistantEl = null;
                break;
            case 'error':
                addMessage('error', msg.content);
                break;
        }
    });
</script>
</body>
</html>`;
    }
}
