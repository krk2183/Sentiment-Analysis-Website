// When you want to make a sub-object you can call MAINOBJECT.call(prop1,prop2) and you need to give the sub-object those same properties so they work
// It's like not having to use this.prop1 = prop1 and saving time 

// You can do Object.setPropertyOf(object1,object2) so when you add a propertial inheritence to object1 object 2 also gets it

// if (!new.target) throw Error('some text')

const library = [];  // This is where we will store the books

function Book(title,author,page_count) { // Creating the main Object we will be using
    this.bookTitle = title;
    this.id = crypto.randomUUID();
    this.bookAuthor = author;
    this.page_count = page_count;
    this.read = false;
}

Book.prototype.remove = function() {  // A prototype function
    const index = library.findIndex(book => book.id === this.id);
    if (index !==-1) {
        library.splice(index,1)
    }
};

// The main function of the program that will handle Element manupilation
// and getting data from the textboxes on the website
function displayBooks() {
    const bookscontainer = document.getElementById('books-container');
    bookscontainer.innerHTML = "";
    
    for (let i=0; i<library.length; i++) {
        // Creating variables for elements so we can easily change their
        // text content (innerHTML)
        const book = library[i];
        const newdiv = document.createElement("div");
        const titledev = document.createElement("div");
        const authordiv = document.createElement("div");
        const pagesdiv = document.createElement("div");
        const iddiv = document.createElement("div");
        const readdiv = document.createElement("div");
        const readbutton = document.createElement("div");
        const removebutton = document.createElement("div");

        newdiv.classList.add("library-div");

        titledev.innerHTML = "Title: " + book.bookTitle;
        authordiv.innerHTML = "Author: " + book.bookAuthor;
        pagesdiv.innerHTML = "Pages: " + book.page_count;
        iddiv.innerHTML = "ID: " + book.id;
        readdiv.innerHTML = "Read: " + (book.read ? "✅" : "❌");

        readbutton.innerHTML = "Toggle Read";
        readbutton.onclick = function() {
            book.read = !book.read;
            displayBooks()
        }

        removebutton.innerHTML = "Remove";
        removebutton.onclick = function() {
            book.remove();
            displayBooks();
        };

        newdiv.appendChild(titledev);
        newdiv.appendChild(authordiv);
        newdiv.appendChild(pagesdiv);
        newdiv.appendChild(iddiv);
        newdiv.appendChild(readdiv);
        newdiv.appendChild(readbutton);
        newdiv.appendChild(removebutton);

        bookscontainer.appendChild(newdiv)
    };

}

document.getElementById('addBookButton').onclick = function() {
    const authorElement = document.getElementById("authorInput");
    const titleElement = document.getElementById("titleInput");
    const pagesElement = document.getElementById("pagesInput");
    // We split these into two pieces because the variable author
    // is a copy of the value in the input field not the input
    // itself whic prevents us from simply setting 'author' to ' '
    // to clear out the field after creating the book
    const author = authorElement.value;
    const title = titleElement.value;
    const pages = pagesElement.value;

    if (author && title && pages) {
        library.push(new Book(title, author, pages));
        
        displayBooks(); 
        
        console.log(library);
        console.log("update");
        

        authorElement.value = '';
        titleElement.value = '';
        pagesElement.value = '';
    } else {
        alert("Please fill in all fields.");
    }
};


