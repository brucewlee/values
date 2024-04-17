let currentDataSet = [];
let currentIndex = 0;
let selectedKey;

// Function to switch between sections
function switchSection(sectionId) {
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('visible');
    });
    document.getElementById(sectionId).classList.add('visible');
}


function populateDataSetSelector() {
    const selector = document.getElementById('datasetSelector');
    for (const key in dataSets) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key;
        selector.appendChild(option);
    }
}

function loadSelectedDataSet() {
    const selector = document.getElementById('datasetSelector');
    selectedKey = selector.value;
    currentDataSet = dataSets[selectedKey];
    currentIndex = 0;
    displayEntry();
}

function displayEntry() {
    if (currentDataSet.length === 0 || currentIndex < 0 || currentIndex >= currentDataSet.length) {
        document.getElementById('entry').innerHTML = "<p>No entry to display.</p>";
        document.getElementById('progress-bar').style.width = '0%';
        document.getElementById('progress-text').innerText = '';
        return;
    }
    const entry = currentDataSet[currentIndex];
    const prompt = entry.Prompt.replace(/\n/g, "<br>");
    const response = entry.Response.replace(/\n/g, "<br>");
    const questionIndex = entry.Question_idx; // Or any identifier used for navigation

    // Assuming 'run_50_1_gpt-3.5-turbo-1106' is derived from 'Question_idx' or similar
    const terminalOutputUrl = `../runs/${selectedKey}/ask_question_terminal.html`;

    document.getElementById('entry').innerHTML = `
        <h2>Entry ${currentIndex + 1}</h2>
        <p><strong>Prompt:</strong> ${prompt}</p>
        <p><strong>Question Index:</strong> ${questionIndex}</p>
        <p><strong>Persona Index:</strong> ${entry.Persona_idx}</p>
        <p><strong>Response:</strong> ${response}</p>
        <p><strong>Response Parsed:</strong> ${entry.Response_Parsed}</p>
    `;

    // Update the navigation div to include the new button
    document.querySelector('.navigation').innerHTML = `
        <button onclick="navigate(-1)"><<<-Previous</button>
        <button onclick="navigate(1)">Next->>></button>
        <button class="button rightbutton" onclick="switchSection('mainNavigator')">Back to Main</button>
        <button class="button rightbutton" onclick="location.href='https://htmlpreview.github.io/?https://github.com/brucewlee/values/blob/main/runs/${selectedKey}/ask_question_terminal.html'">View Terminal Output for This Run</button>
    `;

    // Update progress bar and text
    const progressPercentage = ((currentIndex + 1) / currentDataSet.length) * 100;
    document.getElementById('progress-bar').style.width = `${progressPercentage}%`;
    document.getElementById('progress-text').innerText = `Entry ${currentIndex + 1} of ${currentDataSet.length}`;
}


function navigate(direction) {
    currentIndex += direction;
    if (currentIndex < 0) currentIndex = 0;
    if (currentIndex >= currentDataSet.length) currentIndex = currentDataSet.length - 1;
    displayEntry();
}

populateDataSetSelector();
loadSelectedDataSet();