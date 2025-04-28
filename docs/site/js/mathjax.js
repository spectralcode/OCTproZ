window.MathJax = {
    tex: {
      inlineMath: [["\\(", "\\)"]],
      displayMath: [["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex"
    }
  };
  
  document$.subscribe(() => { 
    MathJax.startup.output.clearCache()
    MathJax.typesetClear()
    MathJax.texReset()
    MathJax.typesetPromise()
  })


  document.addEventListener("DOMContentLoaded", function () {
    if (window.location.pathname.includes("/developer/")) {
      document.body.classList.add("developer-guide");
    }
  });
  

  document.addEventListener("DOMContentLoaded", function () {
    var siteName = document.querySelector('.md-header__topic .md-ellipsis');
    if (document.body.classList.contains('developer-guide')) {
      siteName.textContent = 'OCTproZ Developer Guide';
    } else {
      siteName.textContent = 'OCTproZ User Guide';
    }
  });
  