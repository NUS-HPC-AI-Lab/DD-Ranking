// Populate the sidebar
//
// This is a script, and not included directly in the page, to control the total size of the book.
// The TOC contains an entry for each page, so if each page includes a copy of the TOC,
// the total size of the page becomes O(n**2).
class MDBookSidebarScrollbox extends HTMLElement {
    constructor() {
        super();
    }
    connectedCallback() {
        this.innerHTML = '<ol class="chapter"><li class="chapter-item expanded affix "><a href="introduction.html">Introduction</a></li><li class="chapter-item expanded affix "><a href="contributing.html">Contributing</a></li><li class="chapter-item expanded affix "><li class="part-title">Getting Started</li><li class="chapter-item expanded "><a href="getting-started/installation.html"><strong aria-hidden="true">1.</strong> Installation</a></li><li class="chapter-item expanded "><a href="getting-started/quick-start.html"><strong aria-hidden="true">2.</strong> Quick Start</a></li><li class="chapter-item expanded affix "><li class="part-title">Reference Guide</li><li class="chapter-item expanded "><a href="metrics/overview.html"><strong aria-hidden="true">3.</strong> Metrics</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="metrics/lrs-hard-label.html"><strong aria-hidden="true">3.1.</strong> LabelRobustScoreHard</a></li><li class="chapter-item expanded "><a href="metrics/lrs-soft-label.html"><strong aria-hidden="true">3.2.</strong> LabelRobustScoreSoft</a></li><li class="chapter-item expanded "><a href="metrics/ars.html"><strong aria-hidden="true">3.3.</strong> AugmentationRobustScore</a></li><li class="chapter-item expanded "><a href="metrics/general.html"><strong aria-hidden="true">3.4.</strong> GeneralEvaluator</a></li></ol></li><li class="chapter-item expanded "><a href="augmentations/overview.html"><strong aria-hidden="true">4.</strong> Augmentations</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="augmentations/dsa.html"><strong aria-hidden="true">4.1.</strong> DSA</a></li><li class="chapter-item expanded "><a href="augmentations/cutmix.html"><strong aria-hidden="true">4.2.</strong> CutMix</a></li><li class="chapter-item expanded "><a href="augmentations/mixup.html"><strong aria-hidden="true">4.3.</strong> Mixup</a></li></ol></li><li class="chapter-item expanded "><a href="models/overview.html"><strong aria-hidden="true">5.</strong> Models</a></li><li><ol class="section"><li class="chapter-item expanded "><a href="models/convnet.html"><strong aria-hidden="true">5.1.</strong> ConvNet</a></li><li class="chapter-item expanded "><a href="models/alexnet.html"><strong aria-hidden="true">5.2.</strong> AlexNet</a></li><li class="chapter-item expanded "><a href="models/resnet.html"><strong aria-hidden="true">5.3.</strong> ResNet</a></li><li class="chapter-item expanded "><a href="models/lenet.html"><strong aria-hidden="true">5.4.</strong> LeNet</a></li><li class="chapter-item expanded "><a href="models/vgg.html"><strong aria-hidden="true">5.5.</strong> VGG</a></li><li class="chapter-item expanded "><a href="models/mlp.html"><strong aria-hidden="true">5.6.</strong> MLP</a></li></ol></li><li class="chapter-item expanded "><a href="datasets/overview.html"><strong aria-hidden="true">6.</strong> Datasets</a></li><li class="chapter-item expanded "><a href="config/overview.html"><strong aria-hidden="true">7.</strong> Config</a></li></ol>';
        // Set the current, active page, and reveal it if it's hidden
        let current_page = document.location.href.toString();
        if (current_page.endsWith("/")) {
            current_page += "index.html";
        }
        var links = Array.prototype.slice.call(this.querySelectorAll("a"));
        var l = links.length;
        for (var i = 0; i < l; ++i) {
            var link = links[i];
            var href = link.getAttribute("href");
            if (href && !href.startsWith("#") && !/^(?:[a-z+]+:)?\/\//.test(href)) {
                link.href = path_to_root + href;
            }
            // The "index" page is supposed to alias the first chapter in the book.
            if (link.href === current_page || (i === 0 && path_to_root === "" && current_page.endsWith("/index.html"))) {
                link.classList.add("active");
                var parent = link.parentElement;
                if (parent && parent.classList.contains("chapter-item")) {
                    parent.classList.add("expanded");
                }
                while (parent) {
                    if (parent.tagName === "LI" && parent.previousElementSibling) {
                        if (parent.previousElementSibling.classList.contains("chapter-item")) {
                            parent.previousElementSibling.classList.add("expanded");
                        }
                    }
                    parent = parent.parentElement;
                }
            }
        }
        // Track and set sidebar scroll position
        this.addEventListener('click', function(e) {
            if (e.target.tagName === 'A') {
                sessionStorage.setItem('sidebar-scroll', this.scrollTop);
            }
        }, { passive: true });
        var sidebarScrollTop = sessionStorage.getItem('sidebar-scroll');
        sessionStorage.removeItem('sidebar-scroll');
        if (sidebarScrollTop) {
            // preserve sidebar scroll position when navigating via links within sidebar
            this.scrollTop = sidebarScrollTop;
        } else {
            // scroll sidebar to current active section when navigating via "next/previous chapter" buttons
            var activeSection = document.querySelector('#sidebar .active');
            if (activeSection) {
                activeSection.scrollIntoView({ block: 'center' });
            }
        }
        // Toggle buttons
        var sidebarAnchorToggles = document.querySelectorAll('#sidebar a.toggle');
        function toggleSection(ev) {
            ev.currentTarget.parentElement.classList.toggle('expanded');
        }
        Array.from(sidebarAnchorToggles).forEach(function (el) {
            el.addEventListener('click', toggleSection);
        });
    }
}
window.customElements.define("mdbook-sidebar-scrollbox", MDBookSidebarScrollbox);
