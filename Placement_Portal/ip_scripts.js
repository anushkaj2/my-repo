document.addEventListener("DOMContentLoaded", function() {
    const checkboxes = document.querySelectorAll('input[name="area_of_interest[]"]');
    const othersCheckbox = document.getElementById("others_checkbox");
    const othersTextbox = document.getElementById("others_textbox");
    const form = document.getElementById("registrationForm");
    
    let checkedCount = 0;

    checkboxes.forEach(function(checkbox) {
        checkbox.addEventListener("change", function() {
            if (this.checked) {
                checkedCount++;
            } else {
                checkedCount--;
            }

            if (checkedCount > 3) {
                this.checked = false;
                checkedCount--;
                alert("You can only select up to 3 areas of interest.");
            }
        });
    });

    othersCheckbox.addEventListener("change", function() {
        if (this.checked) {
            othersTextbox.style.display = "block";
        } else {
            othersTextbox.style.display = "none";
            othersTextbox.value="";
        }
    });

    function validateName() {
        const name = document.getElementById("name").value;
        if (/[^a-zA-Z\s.,'-]/.test(name)) {
            alert("Please enter alphabets only in the Name field.");
            return false;
        }
        return true;
    }

    function validateEmail() {
        const email = document.getElementById("email").value;
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        if (!emailRegex.test(email)) {
            alert("Please enter a valid email address.");
            return false;
        }
        return true;
    }

    function validateDob() {
        const dob = new Date(document.getElementById("dob").value);
        if (dob.getFullYear() < 1940) {
            alert("Please enter a valid date of birth.");
            return false;
        }
        return true;
    }

    function validatePreferredLocation() {
        const preferredLocation = document.getElementById("preferred_location").value;
        if (/[^a-zA-Z\s.,'-]/.test(preferredLocation)) {
            alert("Preferred location should contain alphabets only.");
            return false;
        }
        return true;
    }

    function validateCgpa() {
        const cgpa = parseFloat(document.getElementById("cgpa").value);
        if (isNaN(cgpa) || cgpa < 0 || cgpa > 10) {
            alert("Please enter a CGPA value between 0 and 10.");
            return false;
        }
        return true;
    }

    function validateFile() {
        const fileInput = document.getElementById("cv");
        const file = fileInput.files[0];
        if (file && (file.type !== "application/pdf" || file.size > 5 * 1024 * 1024)) {
            alert("Upload a PDF file less than 5MB.");
            return false;
        }
        return true;
    }

    document.getElementById("name").addEventListener("blur", validateName);
    document.getElementById("email").addEventListener("blur", validateEmail);
    document.getElementById("dob").addEventListener("blur", validateDob);
    document.getElementById("preferred_location").addEventListener("blur", validatePreferredLocation);
    document.getElementById("cgpa").addEventListener("blur", validateCgpa);
    document.getElementById("cv").addEventListener("change", validateFile);

    form.addEventListener("submit", function(event) {
        if (!validateName() || !validateEmail() || !validateDob() || !validatePreferredLocation() || !validateCgpa() || !validateFile()) {
            event.preventDefault(); // Prevent form submission if any validation fails
        }
    });
});
