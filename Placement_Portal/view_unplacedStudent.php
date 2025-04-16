<?php
// Database connection
$servername = "localhost";
$username = "IBAB_Placements";
$password = "placementportal@24";
$dbname = "Placements";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// SQL query to get unplaced students
$sql = "SELECT * FROM Student_info WHERE Registration_No NOT IN (SELECT Registration_No FROM Placed)";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    echo "<table>";
    echo "<tr><th>Registration No</th><th>Name</th><th>Email</th><th>DOB</th><th>Program</th><th>Area of Interest</th><th>Preferred Location</th><th>CGPA</th></tr>";

    while ($row = $result->fetch_assoc()) {
        echo "<tr>";
        echo "<td>" . $row["Registration_No"] . "</td>";
        echo "<td>" . $row["Name"] . "</td>";
        echo "<td>" . $row["Email"] . "</td>";
        echo "<td>" . $row["DOB"] . "</td>";
        echo "<td>" . $row["Program"] . "</td>";
        echo "<td>" . $row["Area_Of_Interest"] . "</td>";
        echo "<td>" . $row["Preferred_Location"] . "</td>";
        echo "<td>" . $row["CGPA"] . "</td>";
        echo "</tr>";
    }

    echo "</table>";
} else {
    echo "No entries yet in Student_info table.";
}

$conn->close();
?>
