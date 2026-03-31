import * as vscode from 'vscode';
import { ChatPanel } from './chatPanel';

interface CommandDef {
    id: string;
    prompt: string;
    requiresSelection: boolean;
}

const COMMANDS: CommandDef[] = [
    { id: 'omniscient.explain', prompt: 'Explain this code:', requiresSelection: true },
    { id: 'omniscient.fix', prompt: 'Fix the bug in this code:', requiresSelection: true },
    { id: 'omniscient.test', prompt: 'Write unit tests for this code:', requiresSelection: true },
    { id: 'omniscient.review', prompt: 'Review this code for issues:', requiresSelection: true },
    { id: 'omniscient.doc', prompt: 'Generate documentation for this code:', requiresSelection: true },
    { id: 'omniscient.debug', prompt: 'Debug this error:', requiresSelection: false },
    { id: 'omniscient.commit', prompt: 'Generate a commit message for these changes:', requiresSelection: false },
    { id: 'omniscient.ask', prompt: '', requiresSelection: false },
];

export function registerAgentCommands(
    context: vscode.ExtensionContext,
    getChatPanel: () => ChatPanel
) {
    for (const cmd of COMMANDS) {
        context.subscriptions.push(
            vscode.commands.registerCommand(cmd.id, async () => {
                const editor = vscode.window.activeTextEditor;
                let message = cmd.prompt;

                if (cmd.requiresSelection) {
                    if (!editor || editor.selection.isEmpty) {
                        vscode.window.showWarningMessage(
                            'Please select code first, then run this command.'
                        );
                        return;
                    }
                }

                if (cmd.id === 'omniscient.ask') {
                    const input = await vscode.window.showInputBox({
                        prompt: 'Ask OmniscientLLM a question',
                        placeHolder: 'How do I...',
                    });
                    if (!input) return;
                    message = input;
                }

                if (cmd.id === 'omniscient.commit') {
                    // Get git diff for commit message generation
                    try {
                        const { exec } = require('child_process');
                        const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
                        if (workspaceFolder) {
                            const diff = await new Promise<string>((resolve, reject) => {
                                exec(
                                    'git diff --cached',
                                    { cwd: workspaceFolder, maxBuffer: 1024 * 100 },
                                    (err: any, stdout: string) => {
                                        if (err) reject(err);
                                        else resolve(stdout);
                                    }
                                );
                            });
                            if (diff) {
                                message += `\n\n\`\`\`diff\n${diff.slice(0, 3000)}\n\`\`\``;
                            }
                        }
                    } catch {
                        // Fall through without diff
                    }
                }

                const panel = getChatPanel();
                await panel.sendMessage(message);
            })
        );
    }
}
