document.getElementById("viewUnplacedBtn").addEventListener("click", function() {
    fetch("view_unplacedStudent.php")
        .then(response => response.text())
        .then(data => {
            document.getElementById("unplacedStudentsTable").innerHTML = data;
        })
        .catch(error => console.error("Error fetching unplaced students:", error));
});

document.getElementById("viewPlacedBtn").addEventListener("click", function() {
    fetch("view_placedStudent.php")
        .then(response => response.text())
        .then(data => {
            document.getElementById("placedStudentsTable").innerHTML = data;
        })
        .catch(error => console.error("Error fetching placed students:", error));
});

document.getElementById("resumeForm").addEventListener("submit", function(event) {
    event.preventDefault();
    
    const registrationNo = document.getElementById("registrationNo").value;
    
    fetch(`view_resume.php?registrationNo=${encodeURIComponent(registrationNo)}`)
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById("resumeContent").innerHTML = `
                    <iframe src="data:application/pdf;base64,${data.resume}" type="application/pdf"></iframe>
                    <a href="data:application/pdf;base64,${data.resume}" download="${registrationNo}_resume.pdf">Download Resume</a>
                `;
            } else {
                document.getElementById("resumeContent").innerHTML = "The entered Registration No. does not exist.";
            }
        })
        .catch(error => console.error("Error fetching resume:", error));
});

document.getElementById("checkStudentForm").addEventListener("submit", function(event) {
    event.preventDefault();

    const registrationNo = document.getElementById("registrationNoCheck").value;

    fetch("view_existingPlaced.php", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({ registrationNo: registrationNo })
    })
    .then(response => response.json())
    .then(data => {
        if (data.exists) {
            document.getElementById("existingEntryMessage").textContent = `Student with Registration Number ${registrationNo} already exists in the Placed table.`;
            document.getElementById("existingEntrySection").style.display = "block";
            document.getElementById("addPlacedFormSection").style.display = "none";
            document.getElementById("modifyEntrySection").style.display = "none";
        } else {
            // Check if the student exists in the Student_info table
            fetch("view_existingStudentinfo.php", {
                method: "POST",
                headers: {
                    "Content-Type": "application/x-www-form-urlencoded"
                },
                body: new URLSearchParams({ registrationNo: registrationNo })
            })
            .then(response => response.json())
            .then(infoData => {
                if (infoData.exists) {
                    document.getElementById("addPlacedFormSection").style.display = "block";
                    document.getElementById("existingEntrySection").style.display = "none";
                    document.getElementById("modifyEntrySection").style.display = "none";
                } else {
                    document.getElementById("addPlacedMessage").innerHTML = `The registration number ${registrationNo} does not exist in Student_info table.`;
                    document.getElementById("addPlacedFormSection").style.display = "none";
                }
            })
            .catch(error => console.error("Error checking student in Student_info:", error));
        }
    })
    .catch(error => console.error("Error checking placed student:", error));
});

document.getElementById("removeEntryBtn").addEventListener("click", function() {
    const registrationNo = document.getElementById("registrationNoCheck").value;

    fetch("view_removePlaced.php", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({ registrationNo: registrationNo })
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById("addPlacedMessage").innerHTML = data;
        document.getElementById("existingEntrySection").style.display = "none";
    })
    .catch(error => console.error("Error removing placed student:", error));
});

document.getElementById("modifyEntryBtn").addEventListener("click", function() {
    document.getElementById("modifyEntrySection").style.display = "block";
});

document.getElementById("submitModificationBtn").addEventListener("click", function() {
    const registrationNo = document.getElementById("registrationNoCheck").value;
    const newCompany = document.getElementById("newCompany").value;
    const newPosition = document.getElementById("newPosition").value;
    const newSalary = document.getElementById("newSalary").value;


    fetch("view_modifyPlaced.php", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({
            registrationNo: registrationNo,
            company: newCompany,
            position: newPosition,
            salary: newSalary
        })
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById("addPlacedMessage").innerHTML = data;
        document.getElementById("modifyEntrySection").style.display = "none";
        document.getElementById("existingEntrySection").style.display = "none";
    })
    .catch(error => console.error("Error modifying placed student:", error));
});

document.getElementById("addPlacedBtn").addEventListener("click", function() {
    const registrationNo = document.getElementById("registrationNoCheck").value;
    const company = document.getElementById("company").value;
    const position = document.getElementById("position").value;
    const salary = document.getElementById("salary").value;

    fetch("view_addPlaced.php", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded"
        },
        body: new URLSearchParams({
            registrationNo: registrationNo,
            company: company,
            position: position,
            salary: salary
        })
    })
    .then(response => response.text())
    .then(data => {
        document.getElementById("addPlacedMessage").innerHTML = data;
        document.getElementById("addPlacedFormSection").style.display = "none";
    })
    .catch(error => console.error("Error adding placed student:", error));
});
