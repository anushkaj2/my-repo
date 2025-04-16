<?php
header('Content-Type: text/html');

// Database connection
$servername = "localhost";
$username = "IBAB_Placements";
$password = "placementportal@24";
$dbname = "Placements";

$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

// SQL query to get all placed students
$sql = "SELECT * FROM Placed";
$result = $conn->query($sql);

if ($result->num_rows > 0) {
    echo "<table>";
    echo "<tr><th>Registration No</th><th>Name</th><th>Program</th><th>Company</th><th>Position</th><th>Salary</th></tr>";

    while ($row = $result->fetch_assoc()) {
        echo "<tr>";
        echo "<td>" . $row["Registration_No"] . "</td>";
        echo "<td>" . $row["Name"] . "</td>";
        echo "<td>" . $row["Program"] . "</td>";
        echo "<td>" . $row["Company"] . "</td>";
        echo "<td>" . $row["Position"] . "</td>";
        echo "<td>" . $row["Salary"] . "</td>";
        echo "</tr>";
    }

    echo "</table>";
} else {
    echo "No entries yet in Placed table.";
}

$conn->close();
?>
