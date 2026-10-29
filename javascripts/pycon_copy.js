/**
 * Copy handler for Python console (pycon) code blocks.
 * Strips `>>>`/`...` prompts; comments out REPL output lines with `#` so
 * pasted text stays runnable while keeping the output visible.
 */
document.addEventListener("DOMContentLoaded", function () {
  const COPY_BUTTON_SELECTOR = ".md-clipboard, .md-code__button";

  function primeCopyButton(event) {
    const copyButton = event.target.closest(COPY_BUTTON_SELECTOR);
    if (!copyButton) return;

    const codeBlock = findCodeBlockForCopyButton(copyButton);
    if (!codeBlock) return;

    const rawText = codeBlock.textContent || "";
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    copyButton.setAttribute("data-clipboard-text", stripPythonPrompts(rawText));
    copyButton.removeAttribute("data-clipboard-target");
  }

  function handleSelectionCopy(event) {
    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) return;

    const anchorNode = selection.getRangeAt(0).commonAncestorContainer;
    const codeBlock =
      anchorNode.nodeType === Node.ELEMENT_NODE
        ? anchorNode.closest("code")
        : anchorNode.parentElement?.closest("code");
    if (!codeBlock) return;

    const rawText = selection.toString();
    if (!shouldStripPrompts(codeBlock, rawText)) return;

    event.preventDefault();
    event.clipboardData?.setData("text/plain", stripPythonPrompts(rawText));
  }

  document.addEventListener("click", primeCopyButton, true);
  document.addEventListener("pointerdown", primeCopyButton, true);
  document.addEventListener("copy", handleSelectionCopy, true);
});

function shouldStripPrompts(codeBlock, rawText) {
  return (
    /(^|\n)[ \t]*(>>>|\.\.\.)/.test(rawText) ||
    codeBlock.classList.contains("language-pycon") ||
    codeBlock.closest(".highlight")?.classList.contains("pycon")
  );
}

function findCodeBlockForCopyButton(copyButton) {
  const targetSelector = copyButton.getAttribute("data-clipboard-target");
  if (targetSelector) {
    const target = document.querySelector(targetSelector);
    const targetCode = target?.querySelector?.("code") || target;
    if (targetCode?.tagName?.toLowerCase() === "code") return targetCode;
  }
  return (
    copyButton.closest("pre")?.querySelector("code") ||
    copyButton
      .closest(".highlight, .codehilite, .md-typeset__scrollwrap, .md-typeset")
      ?.querySelector("pre code")
  );
}

/**
 * Best-effort: keeps unprompted lines inside triple-quoted strings as code.
 * Any other unprompted line is REPL output, kept but commented with `#` so
 * the copied text stays valid to paste and run.
 */
function stripPythonPrompts(text) {
  const codeLines = [];
  let inTripleQuotedString = false;

  function toggleTripleQuoteState(sourceLine) {
    const matches = sourceLine.match(/("""|''')/g);
    if (matches && matches.length % 2 === 1) {
      inTripleQuotedString = !inTripleQuotedString;
    }
  }

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();
    if (line.startsWith(">>> ") || line.startsWith("... ")) {
      const stripped = line.slice(4);
      codeLines.push(stripped);
      toggleTripleQuoteState(stripped);
    } else if (line === ">>>" || line === "...") {
      codeLines.push("");
    } else if (inTripleQuotedString) {
      codeLines.push(line);
      toggleTripleQuoteState(line);
    } else if (line === "") {
      codeLines.push("");
    } else {
      codeLines.push(`# ${line}`);
    }
  }

  return codeLines.join("\n").trim();
}
