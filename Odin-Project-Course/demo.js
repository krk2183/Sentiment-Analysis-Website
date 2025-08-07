// Main storage
const myLibrary = [];

// Create the fundemental Book object
function Book(title, author, pages) {
    this.bookTitle = title;
    this.bookAuthor = author;
    this.bookPages = pages;
    this.id = crypto.randomUUID();
    this.read = false;
}

// Prototype remove function
Book.prototype.remove = function () {
    const index = myLibrary.findIndex(book => book.id === this.id);
    if (index !== -1) {
        myLibrary.splice(index, 1);
    }
};

// Site-Display function
function displayBooks() {
    const booksContainer = document.getElementById("books-container"); // Load the container
    booksContainer.innerHTML = ""; // Reset the content inside

    for (let i = 0; i < myLibrary.length; i++) { 
        const book = myLibrary[i];
        // Create variable for each element so you can alter their text content (innerHTML)
        const newDiv = document.createElement("div");
        const titleDiv = document.createElement("div");
        const authorDiv = document.createElement("div");
        const pagesDiv = document.createElement("div");
        const idDiv = document.createElement("div");
        const readDiv = document.createElement("div");
        const readButton = document.createElement("button");
        const removeButton = document.createElement("button");

        newDiv.classList.add("library-div");

        titleDiv.innerHTML = "Title: " + book.bookTitle;
        authorDiv.innerHTML = "Author: " + book.bookAuthor;
        pagesDiv.innerHTML = "Pages: " + book.bookPages;
        idDiv.innerHTML = "ID: " + book.id;
        readDiv.innerHTML = "Read: " + (book.read ? "✅ Yes" : "❌ No");
        readButton.innerHTML = "Toggle Read";
        removeButton.innerHTML = "Remove";

        readButton.onclick = function () {
            book.read = !book.read;
            displayBooks();
        };
        
        removeButton.onclick = function () {
            book.remove();
            displayBooks();
        };

        newDiv.appendChild(titleDiv);
        newDiv.appendChild(authorDiv);
        newDiv.appendChild(pagesDiv);
        newDiv.appendChild(idDiv);
        newDiv.appendChild(readDiv);
        newDiv.appendChild(readButton);
        newDiv.appendChild(removeButton);

        booksContainer.appendChild(newDiv);
    }
}




document.getElementById("addBookButton").onclick = function (){
    const authorInput = document.getElementById("authorInput").value;
    const titleInput = document.getElementById("titleInput").value;
    const pagesInput = document.getElementById("pagesInput").value;
    myLibrary.push(new Book(authorInput, titleInput, pagesInput));
    displayBooks();
    console.log(myLibrary);
    console.log("update");
}