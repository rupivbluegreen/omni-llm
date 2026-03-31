import * as vscode from 'vscode';

export class OmniscientCompletionProvider implements vscode.InlineCompletionItemProvider {
    private debounceTimer: NodeJS.Timeout | undefined;
    private lastRequestTime = 0;
    private readonly debounceMs = 300;

    async provideInlineCompletionItems(
        document: vscode.TextDocument,
        position: vscode.Position,
        context: vscode.InlineCompletionContext,
        token: vscode.CancellationToken
    ): Promise<vscode.InlineCompletionItem[]> {
        // Debounce: skip if called too soon
        const now = Date.now();
        if (now - this.lastRequestTime < this.debounceMs) {
            return [];
        }
        this.lastRequestTime = now;

        // Don't complete in comments or strings (basic heuristic)
        const lineText = document.lineAt(position.line).text;
        const beforeCursor = lineText.substring(0, position.character);
        if (beforeCursor.trimStart().startsWith('//') || beforeCursor.trimStart().startsWith('#')) {
            return [];
        }

        const config = vscode.workspace.getConfiguration('omniscient');
        const serverUrl = config.get<string>('serverUrl', 'http://localhost:8000');

        // Extract prefix and suffix for FIM
        const prefix = document.getText(new vscode.Range(
            new vscode.Position(0, 0),
            position
        ));
        const suffix = document.getText(new vscode.Range(
            position,
            document.lineAt(document.lineCount - 1).range.end
        ));

        // Limit prefix/suffix to avoid huge payloads
        const maxChars = 4000;
        const trimmedPrefix = prefix.slice(-maxChars);
        const trimmedSuffix = suffix.slice(0, maxChars);

        try {
            const controller = new AbortController();
            token.onCancellationRequested(() => controller.abort());

            const response = await fetch(`${serverUrl}/v1/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: trimmedPrefix,
                    suffix: trimmedSuffix,
                    max_tokens: 128,
                    temperature: 0.2,
                }),
                signal: controller.signal,
            });

            if (!response.ok) return [];

            const data = await response.json() as any;
            const completionText = data.choices?.[0]?.text;

            if (!completionText) return [];

            return [
                new vscode.InlineCompletionItem(
                    completionText,
                    new vscode.Range(position, position)
                ),
            ];
        } catch {
            return [];
        }
    }
}
