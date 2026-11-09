/* Top-bar roll-down menus: keyboard and pointer behaviour.
 *
 * CSS alone opens a menu on :hover and :focus-within. This script adds the
 * parts CSS cannot express — click-to-toggle (so touch pointers above the
 * tab-bar breakpoint do not depend on hover emulation), Escape and
 * outside-click dismissal, and an aria-expanded value that tracks every one of
 * those paths.
 */
(function () {
  "use strict";

  function items() {
    return Array.prototype.slice.call(
      document.querySelectorAll(".md-tabs__item--nested")
    );
  }

  function toggleOf(item) {
    return item.querySelector(".md-tabs__toggle");
  }

  function setOpen(item, open) {
    var toggle = toggleOf(item);
    if (!toggle) return;
    item.classList.toggle("md-tabs__item--open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
  }

  function closeAll(except) {
    items().forEach(function (item) {
      if (item !== except) setOpen(item, false);
    });
  }

  function init() {
    var nested = items();
    if (!nested.length) return;

    nested.forEach(function (item) {
      var toggle = toggleOf(item);
      if (!toggle) return;

      toggle.addEventListener("click", function (event) {
        event.preventDefault();
        var open = toggle.getAttribute("aria-expanded") === "true";
        closeAll(item);
        setOpen(item, !open);
      });

      // Keep the attribute honest when CSS opens the menu on hover or focus.
      item.addEventListener("mouseenter", function () {
        setOpen(item, true);
      });
      item.addEventListener("mouseleave", function () {
        setOpen(item, false);
      });
      item.addEventListener("focusin", function () {
        setOpen(item, true);
      });
      item.addEventListener("focusout", function (event) {
        if (!item.contains(event.relatedTarget)) setOpen(item, false);
      });
    });

    document.addEventListener("keydown", function (event) {
      if (event.key !== "Escape") return;
      var open = document.querySelector(".md-tabs__item--open");
      if (!open) return;
      setOpen(open, false);
      var toggle = toggleOf(open);
      if (toggle) toggle.focus();
    });

    document.addEventListener("click", function (event) {
      if (!event.target.closest(".md-tabs__item--nested")) closeAll(null);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
