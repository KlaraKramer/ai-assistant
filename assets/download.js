document.addEventListener("click", function(event) {
    if (event.target && event.target.id === "download-btn") {
        var htmlContent = document.documentElement.outerHTML;
        var blob = new Blob([htmlContent], {type: "text/html"});
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "dashboard_static.html";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
});