import * as vscode from 'vscode';
import { ChatPanel } from './chatPanel';
import { OmniscientCompletionProvider } from './completionProvider';
import { registerAgentCommands } from './agentCommands';

let chatPanel: ChatPanel | undefined;

export function activate(context: vscode.ExtensionContext) {
    console.log('OmniscientLLM extension activated');

    // Register chat panel command
    context.subscriptions.push(
        vscode.commands.registerCommand('omniscient.openChat', () => {
            if (!chatPanel) {
                chatPanel = new ChatPanel(context);
            }
            chatPanel.reveal();
        })
    );

    // Register inline completion provider
    const completionProvider = new OmniscientCompletionProvider();
    context.subscriptions.push(
        vscode.languages.registerInlineCompletionItemProvider(
            { pattern: '**' },
            completionProvider
        )
    );

    // Register slash commands (/explain, /fix, /test, etc.)
    registerAgentCommands(context, () => {
        if (!chatPanel) {
            chatPanel = new ChatPanel(context);
        }
        return chatPanel;
    });
}

export function deactivate() {
    chatPanel?.dispose();
    chatPanel = undefined;
}
