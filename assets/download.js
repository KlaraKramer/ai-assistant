// document.addEventListener("click", function(event) {
//     if (event.target && event.target.id === "download-btn") {
//         var graphs = document.querySelectorAll(".js-plotly-plot");
//         graphs.forEach(function(graph) {
//             Plotly.relayout(graph, {});
//         });

//         setTimeout(() => {
//             var htmlContent = document.documentElement.outerHTML;
//             var blob = new Blob([htmlContent], {type: "text/html"});
//             var a = document.createElement("a");
//             a.href = URL.createObjectURL(blob);
//             a.download = "dashboard_static.html";
//             document.body.appendChild(a);
//             a.click();
//             document.body.removeChild(a);
//         }, 500); // Give some time for the plots to update
//     }
// });