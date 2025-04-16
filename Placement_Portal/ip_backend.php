<?php
$servername = "localhost";
$username = "IBAB_Placements";
$password = "placementportal@24";
$dbname = "Placements";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$registration_no = $_POST['registration_no'];
$name = $_POST['name'];
$email = $_POST['email'];
$dob = $_POST['dob'];
$program = $_POST['program'];


//$area_of_interest = implode(', ', $_POST['area_of_interest']); // Convert array to string


$area_of_interest_string = $_POST['area_of_interest'];

// Check if 'Others' was selected and handle input
if (in_array("Others", $area_of_interest_string) && !empty($_POST['others_value'])) {
    $others_value = $_POST['others_value'];
    // Add "Others" input to the area of interest array
    $area_of_interest_string = array_diff($area_of_interest_string, ["Others"]); // Remove the generic "Others"
    $area_of_interest_string[] = $others_value; // Add the custom value from the text box
}

$area_of_interest = implode(', ', $area_of_interest_string); // Convert array to string





$preferred_location = $_POST['preferred_location'];
$cgpa = $_POST['cgpa'];

// Handle file upload
$cv = $_FILES['cv']['tmp_name'];
$cvContent = file_get_contents($cv);

// SQL query to insert or update with prepared statements
$sql = "INSERT INTO Student_info (registration_no, `name`, email, dob, program, area_of_interest, preferred_location, cgpa, cv)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON DUPLICATE KEY UPDATE 
            `name` = VALUES(`name`),
            email = VALUES(email),
            dob = VALUES(dob),
            program = VALUES(program),
            area_of_interest = VALUES(area_of_interest),
            preferred_location = VALUES(preferred_location),
            cgpa = VALUES(cgpa),
            cv = VALUES(cv)";

// Prepare the statement
$stmt = $conn->prepare($sql);

// Bind parameters to the query
$stmt->bind_param("sssssssss", $registration_no, $name, $email, $dob, $program, $area_of_interest, $preferred_location, $cgpa, $cvContent);

// Execute the query
if ($stmt->execute()) {
    // Redirect to the success page
    header("Location: successfulSubmission");
    exit();
} else {
    echo "Error: " . $stmt->error;
}

// Close the statement and connection
$stmt->close();
$conn->close();

?>
