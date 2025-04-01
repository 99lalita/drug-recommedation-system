function getRecommendations() {
  let user_input = document.getElementById("user_input").value;

  fetch("http://127.0.0.1:5000/recommend", {
    // ✅ Fix: Ensure correct API URL
    method: "POST",
    body: new URLSearchParams({ user_input: user_input }),
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  })
    .then((response) => response.json())
    .then((data) => {
      let output = "<h3>Recommended Drugs:</h3><ul>";
      data.forEach((drug) => {
        output += `<li><strong>${drug.drug_name}</strong> (Condition: ${drug.condition}, Rating: ${drug.rating}) - ${drug.sentiment}</li>`;
      });
      output += "</ul>";
      document.getElementById("recommendations").innerHTML = output;
    })
    .catch((error) => console.error("Error fetching recommendations:", error));
}
