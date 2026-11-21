document.addEventListener("DOMContentLoaded", function () {

  // Favicons served via DuckDuckGo's icon proxy. Hot-linking directly
  // from claude.ai / chatgpt.com is unreliable: both sit behind
  // Cloudflare bot-mitigation, which can challenge or block the
  // request when the browser fetches the icon cross-origin from our
  // docs site. The DuckDuckGo proxy returns the same favicons over
  // plain HTTPS with no bot wall.
  var AI_PROVIDERS = [
    {
      name: "Claude",
      icon: "https://icons.duckduckgo.com/ip3/claude.ai.ico",
      buildUrl: function (prompt) {
        return (
          "https://claude.ai/new?q=" + encodeURIComponent(prompt)
        );
      },
    },
    {
      name: "ChatGPT",
      icon: "https://icons.duckduckgo.com/ip3/chatgpt.com.ico",
      buildUrl: function (prompt) {
        return (
          "https://chatgpt.com/?q=" + encodeURIComponent(prompt)
        );
      },
    },
  ];

  function getPageContent() {
    var article = document.querySelector("article.md-content__inner");
    if (!article) return document.title;
    var clone = article.cloneNode(true);
    // Remove mkdocstrings source code sections (collapsible details blocks)
    clone.querySelectorAll("details").forEach(function (el) {
      var summary = el.querySelector("summary");
      if (summary && /source code/i.test(summary.textContent)) {
        el.remove();
      }
    });
    var removeSelectors = [
      ".md-source-file",
      ".headerlink",
      ".md-annotation",
      "script",
      "style",
    ];
    removeSelectors.forEach(function (sel) {
      clone.querySelectorAll(sel).forEach(function (el) {
        el.remove();
      });
    });
    var text = clone.textContent || clone.innerText || "";
    // Clean up line number sequences (e.g. "21 22 23 24 ...")
    text = text.replace(/(\d+\s+){5,}/g, " ");
    text = text.replace(/\s+/g, " ").trim();
    if (text.length > 4000) {
      text = text.substring(0, 4000) + "...";
    }
    return text;
  }

  function buildPrompt(pageContent) {
    return (
      "Please summarize and explain the following documentation page " +
      'from the Synalinks framework in a clear and concise way. The page is titled "' +
      document.title +
      '".\n\n' +
      "Reference: " +
      window.location.href +
      "\n\n" +
      "Here is the page content:\n\n" +
      pageContent
    );
  }

  function createButton() {
    var container = document.createElement("div");
    container.className = "summarize-ai-container";

    var button = document.createElement("button");
    button.className = "summarize-ai-btn";
    button.setAttribute("aria-label", "Summarize");
    button.setAttribute("title", "Summarize");
    button.innerHTML =
      "<span>Summarize</span>" +
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="22" height="22" fill="currentColor">' +
      '<path d="M7.5 5.6 10 7 8.6 4.5 10 2 7.5 3.4 5 2l1.4 2.5L5 7zm12 9.8L17 14l1.4 2.5L17 19l2.5-1.4L22 19l-1.4-2.5L22 14zM22 2l-2.5 1.4L17 2l1.4 2.5L17 7l2.5-1.4L22 7l-1.4-2.5zm-7.63 5.29c-.39-.39-1.02-.39-1.41 0L1.29 18.96c-.39.39-.39 1.02 0 1.41l2.34 2.34c.39.39 1.02.39 1.41 0L16.7 11.05c.39-.39.39-1.02 0-1.41zm-1.03 5.49-2.12-2.12 2.44-2.44 2.12 2.12z"/>' +
      "</svg>";

    var dropdown = document.createElement("div");
    dropdown.className = "summarize-ai-dropdown";
    dropdown.style.display = "none";

    AI_PROVIDERS.forEach(function (provider) {
      var option = document.createElement("a");
      option.className = "summarize-ai-option";
      option.href = "#";
      option.innerHTML =
        '<img src="' +
        provider.icon +
        '" alt="' +
        provider.name +
        '" width="18" height="18" />' +
        "<span>" +
        provider.name +
        "</span>";
      option.addEventListener("click", function (e) {
        e.preventDefault();
        e.stopPropagation();
        var content = getPageContent();
        var prompt = buildPrompt(content);
        var url = provider.buildUrl(prompt);
        window.open(url, "_blank");
        dropdown.style.display = "none";
      });
      dropdown.appendChild(option);
    });

    button.addEventListener("click", function (e) {
      e.stopPropagation();
      var isVisible = dropdown.style.display === "flex";
      dropdown.style.display = isVisible ? "none" : "flex";
    });

    document.addEventListener("click", function () {
      dropdown.style.display = "none";
    });

    container.appendChild(button);
    container.appendChild(dropdown);
    return container;
  }

  function insertButton() {
    if (document.querySelector(".summarize-ai-container")) return;
    var btn = createButton();
    // Place it in the header, right after the search box.
    var headerInner = document.querySelector(".md-header__inner");
    var search = headerInner && headerInner.querySelector(".md-search");
    if (search && search.parentNode) {
      search.parentNode.insertBefore(btn, search.nextSibling);
    } else {
      // Fallback: float it if the header layout isn't available.
      btn.classList.add("summarize-ai-container--floating");
      document.body.appendChild(btn);
    }
  }

  insertButton();

  if (typeof document$ !== "undefined") {
    document$.subscribe(function () {
      if (!document.querySelector(".summarize-ai-container")) {
        insertButton();
      }
    });
  }
});
