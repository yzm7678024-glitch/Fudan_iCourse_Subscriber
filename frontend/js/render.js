/**
 * iCourse Markdown + KaTeX renderer.
 * Protect code first, identify only well-formed math, then render
 * exclusively the math we explicitly identified.
 */
window.ICS = window.ICS || {};

var _FORMULA_PLACEHOLDER_PREFIX = "\uE000";
var _FORMULA_PLACEHOLDER_SUFFIX = "\uE001";

function _stashFormulas(mdText) {
  var formulas = [];
  var codeParts = [];

  function stash(formula) {
    var key = _FORMULA_PLACEHOLDER_PREFIX + formulas.length
      + _FORMULA_PLACEHOLDER_SUFFIX;
    formulas.push(formula);
    return key;
  }

  function saveCode(code) {
    var key = "\uE010" + codeParts.length + "\uE011";
    codeParts.push(code);
    return key;
  }

  // Protect fenced code blocks, including shell commands and assembly.
  var lines = String(mdText).split("\n");
  var parts = [];
  for (var i = 0; i < lines.length;) {
    var opening = lines[i].match(/^ {0,3}(`{3,}|~{3,})/);
    if (!opening) {
      parts.push(lines[i++]);
      continue;
    }
    var fence = opening[1];
    var closing = new RegExp(
      "^ {0,3}" + fence[0] + "{" + fence.length + ",}[ \\t]*$"
    );
    var end = i + 1;
    while (end < lines.length && !closing.test(lines[end])) end++;
    if (end < lines.length) end++;
    parts.push(saveCode(lines.slice(i, end).join("\n")));
    i = end;
  }
  var text = parts.join("\n");

  // Protect backtick-delimited inline code.
  text = text.replace(/(`+)([^\n]*?)\1/g, function (m) {
    return saveCode(m);
  });

  // Protect bare assembly immediates (both $1, %rdx and $0, (%rsp)).
  text = text.replace(
    /\$-?(?:0x[0-9a-fA-F]+|\d+)(?=\s*,)/g,
    function (m) { return saveCode(m); }
  );

  // Existing parenthesized inline math: no multiline matches.
  text = text.replace(/\\\(([^\n]*?)\\\)/g, function (m, body) {
    if (/[\u3400-\u9fff]/.test(body)) return m;
    return stash(m);
  });

  // Existing bracketed display math: do not swallow prose or headings.
  text = text.replace(/\\\[([\s\S]*?)\\\]/g, function (m, body) {
    if (body.length > 4000 || /[\u3400-\u9fff]/.test(body) ||
        /\n\s*\n|\n\s*#{1,6}\s|\n\s*\|/.test(body)) {
      return m;
    }
    return stash(m);
  });

  // $$ delimiters on their own lines.
  text = text.replace(
    /(^|\n)[ \t]*\$\$[ \t]*\n([\s\S]*?)\n[ \t]*\$\$[ \t]*(?=\n|$)/g,
    function (whole, lead, body) {
      if (body.length > 4000 || /[\u3400-\u9fff]/.test(body) ||
          /\n\s*\n|\n\s*#{1,6}\s|\n\s*\|/.test(body)) {
        return whole;
      }
      return lead + stash("\\[" + body + "\\]");
    }
  );

  // $$E=mc^2$$ on one line.
  text = text.replace(/\$\$([^\n$]+?)\$\$/g, function (whole, body) {
    if (/[\u3400-\u9fff]/.test(body)) return whole;
    return stash("\\[" + body + "\\]");
  });

  // Ordinary inline math. Do not span line breaks or parse Chinese prose.
  text = text.replace(
    /(^|[^\\$])\$([^\n$\uE010\uE011]+?)\$(?!\$)/g,
    function (whole, before, body) {
      if (/^\s|\s$|[\u3400-\u9fff]/.test(body) || body.length > 500) {
        return whole;
      }
      return before + stash("\\(" + body + "\\)");
    }
  );

  // Restore the original code before the Markdown parser handles it.
  text = text.replace(/\uE010(\d+)\uE011/g, function (_, n) {
    return codeParts[Number(n)];
  });

  return { text: text, formulas: formulas };
}

function _restoreFormulas(html, formulas) {
  for (var i = 0; i < formulas.length; i++) {
    var safeFormula = formulas[i]
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
    html = html.split(
      _FORMULA_PLACEHOLDER_PREFIX + i + _FORMULA_PLACEHOLDER_SUFFIX
    ).join('<span class="ics-math">' + safeFormula + '</span>');
  }
  return html;
}

function _renderMarkdown(mdText) {
  if (!mdText) return "";
  var stashed = _stashFormulas(mdText);
  var rawHtml = marked.parse(stashed.text, { breaks: true });
  var restored = _restoreFormulas(rawHtml, stashed.formulas);
  return DOMPurify.sanitize(restored);
}

function _activateKaTeX(element) {
  if (typeof renderMathInElement !== "function") return;
  // Only render formula spans we have validated and stashed. Malformed
  // delimiters left in ordinary text must not consume entire paragraphs.
  element.querySelectorAll(".ics-math").forEach(function (node) {
    renderMathInElement(node, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
      throwOnError: false,
    });
  });
}

function _plainSnippet(mdText, maxLen) {
  maxLen = maxLen || 100;
  if (!mdText) return "";
  var text = mdText
    .replace(/\$\$.+?\$\$/gs, "...")
    .replace(/\\\[.+?\\\]/gs, "...")
    .replace(/\$[^$]+?\$/g, "...")
    .replace(/\\\(.+?\\\)/g, "...")
    .replace(/#{1,6}\s+/g, "")
    .replace(/\*{1,3}(.+?)\*{1,3}/g, "$1")
    .replace(/`{1,3}[^`]*`{1,3}/g, "")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/[|:\-]+/g, " ")
    .replace(/\n+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  return text.length > maxLen ? text.slice(0, maxLen) + "..." : text;
}

window.ICS.render = {
  renderMarkdown: _renderMarkdown,
  activateKaTeX: _activateKaTeX,
  plainSnippet: _plainSnippet,
};
