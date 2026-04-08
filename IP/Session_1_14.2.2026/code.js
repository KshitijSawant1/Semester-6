class Car {
  constructor(brand, model, color, shift) {
    this.brand = brand;
    this.model = model;
    this.color = color;
    this.shift = this.shift;
  }

  displaydetails() {
    console.log();
    console.log(`This is a car ${this.brand} and ${this.model}`);
    console.log(`The car is of ${this.color} and is ${this.shift}`);
  }
}

const car1 = new Car("Toyota", "Camery", "Red", "Manual");

car1.displaydetails();


class Bank {
  constructor(accountHolder, initialAmount) {
    this.accountHolder = accountHolder;
    this.balance = initialAmount;
  }

  deposit(amount) {
    if (amount > 0) {
      this.balance += amount;
      console.log(`₹${amount} deposit done.`);
    } else {
      console.log("Invalid deposit amount.");
    }
  }

  withdraw(amount) {
    if (amount > 0 && amount <= this.balance) {
      this.balance -= amount;
      console.log(`₹${amount} withdrawn done.`);
    } else {
      console.log("Insufficient balance .");
    }
  }

  showBalance() {
    console.log(`Account Holder: ${this.accountHolder}`);
    console.log(`Current Balance: ₹${this.balance}`);
  }
}
const acc1 = new Bank("Kshitij", 5000);

acc1.showBalance();

acc1.deposit(2000);
acc1.showBalance();

acc1.withdraw(3000);
acc1.showBalance();


class Student {
  // Public property
  name;
  
  // Private property: inaccessible outside this class
  #marks = []; 

  constructor(name) {
    this.name = name;
  }

  // Public method to add marks safely
  addMark(mark) {
    if (mark >= 0 && mark <= 100) {
      this.#marks.push(mark);
    } else {
      console.log("Invalid mark");
    }
  }

  // Public method to access private data
  getAverage() {
    if (this.#marks.length === 0) return 0;
    const total = this.#marks.reduce((sum, m) => sum + m, 0);
    return total / this.#marks.length;
  }
}

const student1 = new Student("Alice");
student1.addMark(90);
student1.addMark(85);

console.log(student1.name);      
console.log(student1.getAverage());
// console.log(student1.#marks);   