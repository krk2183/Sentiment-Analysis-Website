const myLibrary = []; // Corrected array name

function Book(title, author, page_count) {
    this.title = title;
    this.id = crypto.randomUUID();
    this.author = author;
    this.page_count = page_count;
    this.read = false;
}

Book.prototype.remove = function() {
    const index = myLibrary.findIndex(book => book.id === this.id);
    if (index !== -1) {
        myLibrary.splice(index, 1);
    }
};

function display() { // The correct function name is 'display'
    const booksContainer = document.getElementById('books-container');
    booksContainer.innerHTML = "";
    
    for (let i = 0; i < myLibrary.length; i++) {
        const book = myLibrary[i];
        
        const newdiv = document.createElement("div");
        const titlediv = document.createElement("div");
        const authordiv = document.createElement("div");
        const pagesdiv = document.createElement("div");
        const iddiv = document.createElement("div");
        const readdiv = document.createElement("div");
        const switchbutton = document.createElement("button");
        const removebutton = document.createElement("button");

        newdiv.classList.add("library-div");

        // Corrected property names to match the constructor
        titlediv.innerHTML = "Title: " + book.title;
        authordiv.innerHTML = "Author: " + book.author;
        pagesdiv.innerHTML = "Pages: " + book.page_count;
        iddiv.innerHTML = "ID: " + book.id;
        readdiv.innerHTML = "Read: " + (book.read ? "✅" : "❌");

        switchbutton.innerHTML = "Switch Read";
        switchbutton.onclick = function() {
            book.read = !book.read;
            display(); // Corrected function call
        };

        removebutton.innerHTML = "Remove";
        removebutton.onclick = function() {
            book.remove();
            display(); // Corrected function call
        };

        newdiv.appendChild(titlediv);
        newdiv.appendChild(authordiv);
        newdiv.appendChild(pagesdiv);
        newdiv.appendChild(iddiv);
        newdiv.appendChild(readdiv);
        newdiv.appendChild(switchbutton);
        newdiv.appendChild(removebutton);

        booksContainer.appendChild(newdiv);
    }
}

document.getElementById('addBookButton').onclick = function() {
    const authorInput = document.getElementById("authortxtbx").value;
    const titleInput = document.getElementById("titletxtbx").value;
    const pagesInput = document.getElementById("pagestxtbx").value;
    
    // Corrected the order of arguments to match the constructor
    myLibrary.push(new Book(titleInput, authorInput, pagesInput));
    display(); // Corrected function call
    
    console.log(myLibrary);
    console.log("update");
};