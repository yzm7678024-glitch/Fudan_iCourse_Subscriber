/**
 * Markdown + KaTeX rendering pipeline.
 * Sets window.ICS.render global.
 *
 * Depends on CDN globals: marked, DOMPurify, renderMathInElement
 *
 * Pipeline: stash formulas → marked → restore formulas → DOMPurify → KaTeX
 * We stash formula delimiters before marked.parse() because characters like
 * * and _ inside $...$ LaTeX (e.g. D^*, P_n, \sum_{i=1}) would otherwise
 * be treated as markdown emphasis and break the formula structure.
 */

window.ICS = window.ICS || {};

var _FORMULA_PLACEHOLDER_PREFIX = "";
var _FORMULA_PLACEHOLDER_SUFFIX = "";

/** Replace $...$ and $$...$$ with placeholders so marked won't touch them. */

function _stashFormulas(mdText) {
  var formulas = [];

  function stash(formula) {
    var key = _FORMULA_PLACEHOLDER_PREFIX
      + formulas.length
      + _FORMULA_PLACEHOLDER_SUFFIX;
    formulas.push(formula);
    return key;
  }

  // 先保护代码，避免将汇编指令中的 $ 当作数学公式。
  var codeParts = [];

  function saveCode(code) {
    var key = "\uE010" + codeParts.length + "\uE011";
    codeParts.push(code);
    return key;
  }

  // 1. 保护 Markdown 围栏代码块（``` 或 ~~~）。
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
      "^ {0,3}" + fence[0]
      + "{" + fence.length + ",}[ \\t]*$"
    );

    var end = i + 1;
    while (end < lines.length && !closing.test(lines[end])) {
      end++;
    }
    if (end < lines.length) end++;

    parts.push(saveCode(lines.slice(i, end).join("\n")));
    i = end;
  }

  var text = parts.join("\n");

  // 2. 保护行内代码，如 `addq $1, %rdx`。
  text = text.replace(/(`+)([^\n]*?)\1/g, function (m) {
    return saveCode(m);
  });

  // 3. 兼容未用反引号包裹的汇编立即数。
  text = text.replace(
    /\$-?(?:0x[0-9a-fA-F]+|\d+)(?=\s*,\s*%[a-zA-Z][a-zA-Z0-9]*)/g,
    function (m) {
      return saveCode(m);
    }
  );

  // 4. 保护原本使用 \( \) 和 \[ \] 的公式。
  text = text.replace(/\\\(([^\n]*?)\\\)/g, function (m) {
    return stash(m);
  });

  text = text.replace(/\\\[([\s\S]*?)\\\]/g, function (m) {
    return stash(m);
  });

  // 5. 多行公式：$$ 独占起止行。
  text = text.replace(
    /(^|\n)[ \t]*\$\$[ \t]*\n([\s\S]*?)\n[ \t]*\$\$[ \t]*(?=\n|$)/g,
    function (_, lead, body) {
      return lead + stash("\\[" + body + "\\]");
    }
  );

  // 6. 单行公式：$$E=mc^2$$。
  text = text.replace(/\$\$([^\n$]+?)\$\$/g, function (_, body) {
    return stash("\\[" + body + "\\]");
  });

  // 7. 行内公式不能跨行，也不能跨越代码区域。
  text = text.replace(
    /(^|[^\\$])\$([^\n$\uE010\uE011]+?)\$(?!\$)/g,
    function (whole, before, body) {
      if (/^\s|\s$/.test(body)) return whole;
      return before + stash("\\(" + body + "\\)");
    }
  );

  // 8. 还原代码，让 marked 正常解析代码块与行内代码。
  text = text.replace(/\uE010(\d+)\uE011/g, function (_, n) {
    return codeParts[Number(n)];
  });

  return { text: text, formulas: formulas };
}


  var text = mdText;

  // 1. Stash existing \(...\) and \[...\] (already-LaTeX, protect from double-processing)
  text = text.replace(/(\\\([\s\S]*?\\\))|(\\\[[\s\S]*?\\\])/g, function (m) {
    return stash(m);
  });

  // 2. Stash $$...$$ (must run before $ → to avoid consuming individual $ chars of $$)
  text = text.replace(/\$\$([\s\S]*?)\$\$/g, function (_, f) {
    return stash("\\[" + f + "\\]");
  });

  // 3. Stash $...$ (inline math — non-greedy to pair nearest closing $)
  text = text.replace(/\$([\s\S]+?)\$/g, function (_, f) {
    return stash("\\(" + f + "\\)");
  });

  return { text: text, formulas: formulas };
}

/** Restore stashed formulas in the HTML output after marked.parse(). */

function _restoreFormulas(html, formulas) {
  for (var i = 0; i < formulas.length; i++) {
    var safeFormula = formulas[i]
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    html = html.split(
      _FORMULA_PLACEHOLDER_PREFIX + i + _FORMULA_PLACEHOLDER_SUFFIX
    ).join(safeFormula);
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
  renderMathInElement(element, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\[", right: "\\]", display: true },
      { left: "\\(", right: "\\)", display: false },
      // NOTE: $...$ intentionally omitted from KaTeX — converted to \(...\) in _stashFormulas
    ],
    throwOnError: false,
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
