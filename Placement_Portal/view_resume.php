<?php
header('Content-Type: application/json');

// Database connection
$servername = "localhost";
$username = "IBAB_Placements";
$password = "placementportal@24";
$dbname = "Placements";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die(json_encode(['success' => false, 'message' => "Connection failed: " . $conn->connect_error]));
}

// Get the registration number from query string
$registrationNo = $conn->real_escape_string($_GET['registrationNo']);

// SQL query to get resume
$sql = "SELECT CV FROM Student_info WHERE Registration_No = '$registrationNo'";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    $row = $result->fetch_assoc();
    $resume = base64_encode($row['CV']);
    echo json_encode(['success' => true, 'resume' => $resume]);
} else {
    echo json_encode(['success' => false, 'message' => "No resume found for this student."]);
}

$conn->close();
?>
