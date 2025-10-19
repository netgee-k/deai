class LiveHighlighter {
    constructor() {
        this.colors = {
            'ai_pattern': '#ff6b6b',
            'style_issue': '#ffd93d', 
            'good_match': '#6bcf7f',
            'signature_phrase': '#4ecdc4'
        };
    }

    highlightText(text, analysis) {
        let highlighted = text;
        
        analysis.issues.forEach(issue => {
            if (issue.position !== undefined) {
                const color = this.getColorForIssue(issue.type);
                const highlightedText = `<span class="highlight ${issue.type}" 
                    style="background-color: ${color}; padding: 2px; border-radius: 3px;"
                    title="${issue.reason} - ${issue.suggestion}">${issue.text}</span>`;
                
                highlighted = this.replaceAtPosition(highlighted, issue.position, issue.text, highlightedText);
            }
        });
        
        return highlighted;
    }

    getColorForIssue(issueType) {
        const colorMap = {
            'formal_word': this.colors.ai_pattern,
            'redundant_phrase': this.colors.ai_pattern,
            'ai_cliche': this.colors.ai_pattern,
            'sentence_rhythm': this.colors.style_issue,
            'vocabulary_mismatch': this.colors.style_issue,
            'signature_phrase': this.colors.signature_phrase
        };
        return colorMap[issueType] || this.colors.good_match;
    }

    replaceAtPosition(text, position, original, replacement) {
        return text.substring(0, position) + replacement + text.substring(position + original.length);
    }
}
