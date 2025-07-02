import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, select, func
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Mapped, mapped_column
from typing import List, Optional
import plotly.express as px

# Database Models (SQLAlchemy 2.0)
Base = declarative_base()

class Stream(Base):
    __tablename__ = "stream"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    category: Mapped[str]  # Operate, Execute, Project, Turnaround
    
    # Relationships
    roles: Mapped[List["Role"]] = relationship(
        secondary="stream_role_table", back_populates="streams"
    )
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(
        back_populates="stream"
    )

class Role(Base):
    __tablename__ = "role"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    
    # Relationships
    streams: Mapped[List["Stream"]] = relationship(
        secondary="stream_role_table", back_populates="roles"
    )

class Shift(Base):
    __tablename__ = "shift"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]

class Employee(Base):
    __tablename__ = "employee"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    wopid: Mapped[Optional[str]]
    first_name: Mapped[Optional[str]]
    last_name: Mapped[Optional[str]]
    on_leave: Mapped[bool] = mapped_column(default=False)
    is_facility_trainer: Mapped[bool] = mapped_column(default=False)
    start_date_KGP_ops: Mapped[Optional[str]]  # Using str for datetime
    
    # Relationships
    allocations: Mapped[List["EmployeeAllocation"]] = relationship(
        back_populates="employee"
    )
    compliance_status: Mapped[List["EmployeeComplianceStatus"]] = relationship(
        back_populates="employee"
    )

class EmployeeAllocation(Base):
    __tablename__ = "employee_allocation"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    employee_id: Mapped[int]
    stream_id: Mapped[int]
    role_id: Mapped[int]
    shift_id: Mapped[int]
    capacity: Mapped[float]
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="allocations")
    stream: Mapped["Stream"] = relationship(back_populates="allocations")

class ComplianceRequirements(Base):
    __tablename__ = "compliance_requirements"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    comp_role_id: Mapped[int]

class EmployeeComplianceStatus(Base):
    __tablename__ = "employee_compliance_status"
    
    id: Mapped[int] = mapped_column(primary_key=True)
    staff_id: Mapped[int]
    comp_req_id: Mapped[int]
    status: Mapped[str]
    
    # Relationships
    employee: Mapped["Employee"] = relationship(back_populates="compliance_status")

# Streamlit App Configuration
st.set_page_config(
    page_title="Employee Compliance Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def init_database():
    """Initialize database connection"""
    # Replace with your actual database URL
    DATABASE_URL = "sqlite:///employee_compliance.db"  # Change this to your DB
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(bind=engine)
    return engine, SessionLocal

@st.cache_data
def get_employee_data(_session):
    """Fetch employee data with compliance status"""
    
    # Query to get employee allocations with related data
    query = select(
        Employee.id.label('employee_id'),
        Employee.first_name,
        Employee.last_name,
        Shift.name.label('shift_name'),
        Role.name.label('role_name'),
        Stream.name.label('stream_name'),
        Stream.category.label('stream_category'),
        EmployeeAllocation.capacity,
        Employee.on_leave,
        Employee.is_facility_trainer
    ).select_from(
        Employee
    ).join(
        EmployeeAllocation, Employee.id == EmployeeAllocation.employee_id
    ).join(
        Stream, EmployeeAllocation.stream_id == Stream.id
    ).join(
        Role, EmployeeAllocation.role_id == Role.id
    ).join(
        Shift, EmployeeAllocation.shift_id == Shift.id
    ).order_by(Employee.first_name, Employee.last_name)
    
    result = _session.execute(query).fetchall()
    
    # Convert to DataFrame
    df = pd.DataFrame(result, columns=[
        'employee_id', 'first_name', 'last_name', 'shift_name', 'role_name', 
        'stream_name', 'stream_category', 'capacity', 'on_leave', 'is_facility_trainer'
    ])
    
    # Create full name
    df['Name'] = df['first_name'].fillna('') + ' ' + df['last_name'].fillna('')
    df['Name'] = df['Name'].str.strip()
    
    return df

@st.cache_data
def get_stream_names(_session):
    """Fetch all stream names from the database"""
    
    query = select(Stream.name).order_by(Stream.name)
    result = _session.execute(query).fetchall()
    
    # Return list of stream names
    return [row[0] for row in result]

@st.cache_data
def get_compliance_data(_session, employee_ids, stream_names):
    """Fetch compliance status data for employees across all streams"""
    
    # Query to get employee compliance status for each stream
    # This joins employee allocations with compliance status
    query = select(
        Employee.id.label('employee_id'),
        Employee.first_name,
        Employee.last_name,
        Stream.name.label('stream_name'),
        EmployeeComplianceStatus.status
    ).select_from(
        Employee
    ).join(
        EmployeeAllocation, Employee.id == EmployeeAllocation.employee_id
    ).join(
        Stream, EmployeeAllocation.stream_id == Stream.id
    ).outerjoin(
        EmployeeComplianceStatus, Employee.id == EmployeeComplianceStatus.staff_id
    ).where(
        Employee.id.in_(employee_ids)
    )
    
    result = _session.execute(query).fetchall()
    
    # Convert to DataFrame
    compliance_df = pd.DataFrame(result, columns=[
        'employee_id', 'first_name', 'last_name', 'stream_name', 'status'
    ])
    
    return compliance_df

def create_compliance_matrix(df, session):
    """Create the compliance matrix similar to your spreadsheet"""
    
    # Get unique employees from the filtered data
    unique_employees = df.groupby(['Name', 'shift_name', 'role_name']).first().reset_index()
    
    # Get all stream names dynamically from database
    stream_names = get_stream_names(session)
    
    # Get employee IDs for compliance lookup
    employee_ids = df['employee_id'].unique() if 'employee_id' in df.columns else []
    
    # Initialize matrix with employee info
    matrix_df = unique_employees[['Name', 'shift_name', 'role_name']].copy()
    
    if len(employee_ids) > 0:
        # Get compliance data
        compliance_df = get_compliance_data(session, employee_ids, stream_names)
        
        # Create employee name mapping
        if not compliance_df.empty:
            compliance_df['full_name'] = (
                compliance_df['first_name'].fillna('') + ' ' + 
                compliance_df['last_name'].fillna('')
            ).str.strip()
            
            # Pivot compliance data to get streams as columns
            compliance_pivot = compliance_df.pivot_table(
                index='full_name',
                columns='stream_name',
                values='status',
                aggfunc='first',
                fill_value=''
            )
            
            # Merge with matrix_df
            matrix_df = matrix_df.merge(
                compliance_pivot,
                left_on='Name',
                right_index=True,
                how='left'
            )
    
    # Ensure all stream columns exist (fill missing streams with empty values)
    for stream_name in stream_names:
        if stream_name not in matrix_df.columns:
            matrix_df[stream_name] = ''
    
    # Convert compliance status to symbols
    def convert_status(status):
        if pd.isna(status) or status == '':
            return ''
        elif status.lower() in ['compliant', 'complete', 'passed', 'yes', 'true']:
            return '✓'
        elif status.lower() in ['pending', 'in_progress', 'partial']:
            return '⚠️'
        elif status.lower() in ['non_compliant', 'failed', 'no', 'false']:
            return '❌'
        else:
            return status
    
    # Apply status conversion to stream columns
    for stream_name in stream_names:
        if stream_name in matrix_df.columns:
            matrix_df[stream_name] = matrix_df[stream_name].apply(convert_status)
    
    return matrix_df, stream_names

def style_compliance_cell(val):
    """Style cells based on compliance status"""
    if val == '✓':
        return 'background-color: #90EE90'  # Light green
    elif val == '⚠️':
        return 'background-color: #FFD700'  # Gold
    elif val == '❌':
        return 'background-color: #FFB6C1'  # Light pink
    else:
        return ''

def main():
    st.title("🏭 Employee Compliance Dashboard")
    st.markdown("---")
    
    # Initialize database
    try:
        engine, SessionLocal = init_database()
        session = SessionLocal()
        
        # Sidebar filters
        st.sidebar.header("Filters")
        
        # Get data
        df = get_employee_data(session)
        
        if df.empty:
            st.warning("No employee data found. Please check your database connection and data.")
            return
        
        # Filter options
        shifts = st.sidebar.multiselect(
            "Select Shifts:",
            options=df['shift_name'].unique(),
            default=df['shift_name'].unique()
        )
        
        roles = st.sidebar.multiselect(
            "Select Roles:",
            options=df['role_name'].unique(),
            default=df['role_name'].unique()
        )
        
        streams = st.sidebar.multiselect(
            "Select Streams:",
            options=df['stream_name'].unique(),
            default=df['stream_name'].unique()
        )
        
        # Apply filters
        filtered_df = df[
            (df['shift_name'].isin(shifts)) &
            (df['role_name'].isin(roles)) &
            (df['stream_name'].isin(streams))
        ]
        
        # Create compliance matrix
        matrix_df, stream_names = create_compliance_matrix(filtered_df, session)
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Employees", len(matrix_df))
        with col2:
            st.metric("Active Shifts", len(shifts))
        with col3:
            st.metric("Roles", len(roles))
        with col4:
            st.metric("Streams", len(stream_names))
        
        st.markdown("---")
        
        # Main compliance matrix
        st.subheader("📋 Employee Compliance Matrix")
        
        # Prepare display dataframe with dynamic stream columns
        base_columns = ['Name', 'shift_name', 'role_name']
        display_columns = base_columns + stream_names
        
        # Ensure all columns exist in matrix_df
        for col in display_columns:
            if col not in matrix_df.columns:
                matrix_df[col] = ''
        
        display_df = matrix_df[display_columns].copy()
        
        # Rename columns for better display
        column_mapping = {
            'shift_name': 'Shift',
            'role_name': 'Role'
        }
        # Keep stream names as they are from the database
        for stream in stream_names:
            column_mapping[stream] = stream
        
        display_df = display_df.rename(columns=column_mapping)
        
        # Style the dataframe - apply styling to all stream columns
        styled_df = display_df.style.applymap(
            style_compliance_cell,
            subset=stream_names  # Apply styling to dynamic stream columns
        )
        
        # Display the matrix
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=600
        )
        
        # Legend
        st.markdown("### 📝 Legend")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("✅ **Compliant** - All requirements met")
        with col2:
            st.markdown("⚠️ **Pending** - In progress or needs attention")
        with col3:
            st.markdown("❌ **Non-compliant** - Requirements not met")
        
        # Additional insights
        st.markdown("---")
        st.subheader("📊 Compliance Insights")
        
        # Create some basic charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Compliance by role
            role_compliance = filtered_df.groupby('role_name').size().reset_index(name='count')
            fig_role = px.bar(
                role_compliance, 
                x='role_name', 
                y='count',
                title="Employees by Role",
                labels={'role_name': 'Role', 'count': 'Number of Employees'}
            )
            st.plotly_chart(fig_role, use_container_width=True)
        
        with col2:
            # Compliance by shift
            shift_compliance = filtered_df.groupby('shift_name').size().reset_index(name='count')
            fig_shift = px.pie(
                shift_compliance,
                values='count',
                names='shift_name',
                title="Employee Distribution by Shift"
            )
            st.plotly_chart(fig_shift, use_container_width=True)
        
        session.close()
        
    except Exception as e:
        st.error(f"Database connection error: {str(e)}")
        st.info("Please ensure your database is properly configured and accessible.")

if __name__ == "__main__":
    main()
